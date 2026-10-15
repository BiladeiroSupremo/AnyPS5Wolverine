#include "SpirvBackend/SpirvMemory/SpirvDescriptors.hpp"
#include "SpirvBackend/SpirvMemory/SpirvTypes.hpp"
#include "SpirvBackend/SpirvMemory/SpirvConstants.hpp"
#include "SpirvBackend/SpirvEmitterHelpers.hpp"
#include "Optimization/ResourceMaterializer.hpp"
#include "Optimization/BindlessHeaps.hpp"
#include "PipelineSpecialization.hpp"
#include <spirv/unified1/spirv.hpp>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace ShaderRecompiler
{
namespace {

constexpr RdnaImageDimensionInfo ImageDimensions[] = {
    {RdnaImageDimension::Dim1D, spv::Dim1D, 1, 1, 0, 0},
    {RdnaImageDimension::Dim2D, spv::Dim2D, 2, 2, 0, 0},
    {RdnaImageDimension::Dim3D, spv::Dim3D, 3, 3, 0, 0},
    //{RdnaImageDimension::DimCube, spv::DimCube, 3, 2, 0, 0},
    {RdnaImageDimension::Dim1DArray, spv::Dim1D, 2, 1, 1, 0},
    {RdnaImageDimension::Dim2DArray, spv::Dim2D, 3, 2, 1, 0},
    {RdnaImageDimension::Dim2DMsaa, spv::Dim2D, 2, 2, 0, 1},
    {RdnaImageDimension::Dim2DMsaaArray, spv::Dim2D, 3, 2, 1, 1},
};

}

[[noreturn]] void ExitDescriptorBindingFailure(const SpirvEmitterState& state, DescriptorBindingKind kind, std::uint32_t resource, const char* reason) {
    throw std::runtime_error("shader binding resolution failed during SPIR-V emit: hash=" + std::to_string(state.program.Resources().shaderHash) + " stage=" + std::to_string(static_cast<std::uint32_t>(StageOf(state))) + " resource=" + std::to_string(resource) + " binding_kind=" + std::to_string(static_cast<std::uint32_t>(kind)) + " reason=" + reason);
}

std::uint32_t ResourceForDescriptor(const SpirvEmitterState& state, DescriptorBindingKind kind, std::uint32_t resource) {
    for (const auto& descriptor : state.program.Metadata().bindings.descriptors) {
        if (descriptor.kind != kind) {
            continue;
        }
        const auto found = std::find(descriptor.resources.begin(), descriptor.resources.end(), resource);
        if (found == descriptor.resources.end()) {
            ExitDescriptorBindingFailure(state, kind, resource, "resource is absent from descriptor group");
        }
        return static_cast<std::uint32_t>(found - descriptor.resources.begin());
    }
    ExitDescriptorBindingFailure(state, kind, resource, "descriptor group was not allocated");
}

std::uint32_t DescriptorElementPointer(SpirvEmitterState& state, std::uint32_t resultPtrType, std::uint32_t variableId, std::uint32_t arrayIndex, DescriptorBindingKind kind, std::uint32_t resource, const char* variableName) {
    if (variableId == 0) {
        ExitDescriptorBindingFailure(state, kind, resource, variableName);
    }
    const auto pointer = state.module.AllocateId();
    const bool heap = kind == DescriptorBindingKind::Samplers || ImageBindingResourceClass(kind) != ImageResourceClass::None;
    const auto index = heap && !state.registrationModules ? state.module.SpecializationConstant(TypeU32(state), PipelineSpecialization::DescriptorIndex(static_cast<std::uint32_t>(kind), arrayIndex), arrayIndex) : ConstantU32(state, arrayIndex);
    state.module.AddFunction(spv::OpAccessChain, resultPtrType, pointer, variableId, index);
    return pointer;
}

std::uint32_t RuntimeImageDword(SpirvEmitterState& state, std::uint32_t resource, std::uint32_t member) {
    const auto index = state.runtimeImageMetadata != 0u ? state.runtimeImageMetadata : ConstantU32(state, resource);
    const auto dword = Binary(state, spv::OpIAdd, TypeU32(state), ConstantU32(state, state.program.Metadata().bindings.ImageMetadataDword() + member), Binary(state, spv::OpIMul, TypeU32(state), index, ConstantU32(state, sizeof(RuntimeAbi::ResourceMetadata) / sizeof(std::uint32_t))));
    const auto pointer = state.module.AllocateId();
    state.module.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state), pointer, state.shaderDataStorageVariable, ConstantU32(state, 0u), dword);
    const auto value = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
    return value;
}

std::uint32_t RuntimeMipCount(SpirvEmitterState& state, std::uint32_t resource) {
    const auto word = RuntimeImageDword(state, resource, offsetof(RuntimeAbi::ResourceMetadata, descriptor) / sizeof(std::uint32_t) + 3u);
    const auto first = EmitAndConstant(state, Binary(state, spv::OpShiftRightLogical, TypeU32(state), word, ConstantU32(state, 12u)), 0xfu);
    const auto last = EmitAndConstant(state, Binary(state, spv::OpShiftRightLogical, TypeU32(state), word, ConstantU32(state, 16u)), 0xfu);
    return Binary(state, spv::OpIAdd, TypeU32(state), Binary(state, spv::OpISub, TypeU32(state), last, first), ConstantU32(state, 1u));
}

std::uint32_t TableImageIndex(SpirvEmitterState& state, DescriptorBindingKind kind, std::uint32_t resource) {
    const bool storage = ImageBindingResourceClass(kind) == ImageResourceClass::Storage;
    const auto supported = [&](std::uint32_t capability) {
        return std::find(state.supportedCapabilities.begin(), state.supportedCapabilities.end(), capability) != state.supportedCapabilities.end();
    };
    const auto dynamic = storage ? spv::CapabilityStorageImageArrayDynamicIndexing : spv::CapabilitySampledImageArrayDynamicIndexing;
    if (!supported(dynamic)) {
        ExitDescriptorBindingFailure(state, kind, resource, "bindless image table needs image array dynamic indexing, which the device lacks");
    }
    state.module.EmitCapability(dynamic);
    if (state.runtimeImageMetadata == 0u) FailEmit("bindless image has no runtime metadata index");
    const auto offset = Binary(state, spv::OpIAdd, TypeU32(state), ConstantU32(state, state.program.Metadata().bindings.ImageMetadataDword() + offsetof(RuntimeAbi::ResourceMetadata, firstElement) / sizeof(std::uint32_t)), Binary(state, spv::OpIMul, TypeU32(state), state.runtimeImageMetadata, ConstantU32(state, sizeof(RuntimeAbi::ResourceMetadata) / sizeof(std::uint32_t))));
    const auto metadata = state.module.AllocateId();
    state.module.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state), metadata, state.shaderDataStorageVariable, ConstantU32(state, 0u), offset);
    const auto index = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, TypeU32(state), index, metadata);
    if (!state.tableIndexNonUniform) {
        return index;
    }
    const auto nonUniform = storage ? spv::CapabilityStorageImageArrayNonUniformIndexing : spv::CapabilitySampledImageArrayNonUniformIndexing;
    const bool extension = state.spirvVersion < 0x00010500u;
    if (!supported(spv::CapabilityShaderNonUniform) || !supported(nonUniform) || (extension && std::find(state.supportedExtensions.begin(), state.supportedExtensions.end(), "SPV_EXT_descriptor_indexing") == state.supportedExtensions.end())) {
        ResourceMaterializer::CountBindlessRejection(BindlessRejection::NonUniform);
        throw std::runtime_error("bindless image table: the slot is not uniform over the workgroup and the device lacks non-uniform image indexing");
    }
    state.module.EmitCapability(spv::CapabilityShaderNonUniform);
    state.module.EmitCapability(nonUniform);
    if (extension) state.module.EmitExtension("SPV_EXT_descriptor_indexing");
    state.module.AddAnnotation(spv::OpDecorate, index, spv::DecorationNonUniform);
    return index;
}

void DecorateTableNonUniform(SpirvEmitterState& state, std::uint32_t slotId, std::uint32_t id) {
    if (slotId != 0 && state.tableIndexNonUniform) state.module.AddAnnotation(spv::OpDecorate, id, spv::DecorationNonUniform);
}

std::uint32_t ImageScalarType(SpirvEmitterState& state, IrTextureNumericClass numericClass) {
    switch (numericClass) {
    case IrTextureNumericClass::Float: return TypeF32(state);
    case IrTextureNumericClass::Uint: return TypeU32(state);
    case IrTextureNumericClass::Sint: return TypeI32(state);
    case IrTextureNumericClass::Unsupported: break;
    }
    FailEmit("invalid image numeric class");
}

std::uint32_t ImageVectorType(SpirvEmitterState& state, IrTextureNumericClass numericClass, std::uint32_t components) {
    return state.module.Type(spv::OpTypeVector, ImageScalarType(state, numericClass), components);
}

std::uint32_t ImageType(SpirvEmitterState& state, const ImageResource& image) {
    std::uint32_t sampled = 0;
    std::uint32_t format = spv::ImageFormatUnknown;
    if (image.resourceClass == ImageResourceClass::Sampled) {
        if (image.atomic) {
            FailEmit("sampled image cannot be atomic");
        }
        sampled = 1;
    } else if (image.resourceClass == ImageResourceClass::Storage) {
        if (image.numericClass == IrTextureNumericClass::Sint || image.numericClass == IrTextureNumericClass::Unsupported) {
            FailEmit("storage image numeric class is not supported");
        }
        sampled = 2;
        if (image.atomic) {
            if (image.numericClass != IrTextureNumericClass::Uint) {
                FailEmit("atomic storage image must be uint");
            }
            format = image.atomic64 ? spv::ImageFormatR64ui : spv::ImageFormatR32ui;
        }
    } else {
        FailEmit("invalid image resource class");
    }
    const auto& info = RdnaImageDimensionInfoFor(image.dimension);
    if (info.multisampled != 0u) {
        if (sampled == 2u) {
            if (std::ranges::find(state.supportedCapabilities, spv::CapabilityStorageImageMultisample) == state.supportedCapabilities.end()) FailEmit("storage image multisampling is unavailable on the target device");
            state.module.EmitCapability(spv::CapabilityStorageImageMultisample);
        }
        if (info.arrayed != 0u) state.module.EmitCapability(spv::CapabilityImageMSArray);
    }
    const auto scalar = image.atomic64 ? TypeScalarU64(state) : ImageScalarType(state, image.numericClass);
    return state.module.Type(spv::OpTypeImage, scalar, info.spirvDimension, image.depthCompare ? 1u : 0u, info.arrayed, info.multisampled, sampled, format);
}

std::uint32_t ImageViewSizeType(SpirvEmitterState& state, RdnaImageDimension dimension) {
    switch (RdnaImageDimensionInfoFor(dimension).coordinateComponents) {
    case 1u: return TypeU32(state);
    case 2u: return TypeU32Vector(state, 2);
    case 3u: return TypeU32Vector(state, 3);
    default: FailEmit("image view size component count is invalid");
    }
}

namespace {

// Descriptor indexing for the heap set: runtime arrays indexed with a non-uniform slot.
void RequireBindlessHeapCapabilities(SpirvEmitterState& state) {
    const auto supported = [&](std::uint32_t capability) {
        return std::find(state.supportedCapabilities.begin(), state.supportedCapabilities.end(), capability) != state.supportedCapabilities.end();
    };
    const bool extension = state.spirvVersion < 0x00010500u;
    if (!supported(spv::CapabilityRuntimeDescriptorArray) || !supported(spv::CapabilityShaderNonUniform) || !supported(spv::CapabilitySampledImageArrayNonUniformIndexing)
        || (extension && std::find(state.supportedExtensions.begin(), state.supportedExtensions.end(), "SPV_EXT_descriptor_indexing") == state.supportedExtensions.end())) {
        FailEmit("bindless image heap: the device lacks runtime descriptor arrays or non-uniform image indexing");
    }
    state.module.EmitCapability(spv::CapabilityRuntimeDescriptorArray);
    state.module.EmitCapability(spv::CapabilityShaderNonUniform);
    state.module.EmitCapability(spv::CapabilitySampledImageArrayNonUniformIndexing);
    if (extension) state.module.EmitExtension("SPV_EXT_descriptor_indexing");
}

// `struct { uint words[]; }`, the block type of the heap's translation and feedback buffers.
std::uint32_t BindlessHeapBlockType(SpirvEmitterState& state) {
    const auto array = state.module.DecoratedType(spv::OpTypeRuntimeArray, {{spv::OpDecorate, {spv::DecorationArrayStride, 4u}}}, TypeU32(state));
    return state.module.DecoratedType(spv::OpTypeStruct, {{spv::OpDecorate, {spv::DecorationBlock}}, {spv::OpMemberDecorate, {0u, spv::DecorationOffset, 0u}}}, array);
}

std::uint32_t DefineBindlessHeapVariable(SpirvEmitterState& state, std::uint32_t type, std::uint32_t storageClass, std::uint32_t binding, const char* name) {
    const auto variable = state.module.DefineGlobalVariable(TypePointer(state, storageClass, type), storageClass);
    state.module.AddName(variable, name);
    state.module.AddAnnotation(spv::OpDecorate, variable, spv::DecorationDescriptorSet, BindlessHeaps::DescriptorSet);
    state.module.AddAnnotation(spv::OpDecorate, variable, spv::DecorationBinding, binding);
    return variable;
}

// The image at heap slot `slot` (a non-uniform index into the kind's runtime array). Each image type
// a mode asks for gets its own variable aliasing the kind's binding.
std::uint32_t LoadBindlessHeapImage(SpirvEmitterState& state, const ImageResource& root, const ImageResource& image, std::uint32_t slot) {
    // The kind follows the table root (HeapEligible tested its dimension), the type the mode.
    const auto kind = BindlessHeapKind(root);
    const auto imageType = ImageType(state, image);
    auto& variable = state.bindlessHeapImageVariables[{kind, imageType}];
    if (variable == 0u) {
        RequireBindlessHeapCapabilities(state);
        static constexpr const char* names[BindlessHeaps::KindCount] = {"bindless_heap_2d", "bindless_heap_2d_array", "bindless_heap_3d", "bindless_heap_1d"};
        variable = DefineBindlessHeapVariable(state, state.module.Type(spv::OpTypeRuntimeArray, imageType), spv::StorageClassUniformConstant, kind, names[kind]);
    }
    const auto pointer = state.module.AllocateId();
    state.module.AddFunction(spv::OpAccessChain, state.module.Type(spv::OpTypePointer, spv::StorageClassUniformConstant, imageType), pointer, variable, slot);
    if (!state.bindlessHeapUniformSlot) state.module.AddAnnotation(spv::OpDecorate, pointer, spv::DecorationNonUniform);
    const auto loaded = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, imageType, loaded, pointer);
    if (!state.bindlessHeapUniformSlot) state.module.AddAnnotation(spv::OpDecorate, loaded, spv::DecorationNonUniform);
    return loaded;
}

}

std::uint32_t BindlessHeapKind(const ImageResource& image) {
    switch (image.dimension) {
    case RdnaImageDimension::Dim2D: return BindlessHeaps::ImageBinding2D;
    case RdnaImageDimension::Dim2DArray: return BindlessHeaps::ImageBinding2DArray;
    case RdnaImageDimension::Dim3D: return BindlessHeaps::ImageBinding3D;
    case RdnaImageDimension::Dim1D: return BindlessHeaps::ImageBinding1D;
    default: FailEmit("bindless image heap: unsupported image dimension");
    }
}

std::uint32_t BindlessHeapBuffer(SpirvEmitterState& state, bool feedback) {
    auto& variable = feedback ? state.bindlessHeapFeedbackVariable : state.bindlessHeapTranslationVariable;
    if (variable == 0u) {
        RequireBindlessHeapCapabilities(state);
        variable = DefineBindlessHeapVariable(state, BindlessHeapBlockType(state), spv::StorageClassStorageBuffer, feedback ? BindlessHeaps::FeedbackBinding : BindlessHeaps::TranslationBinding, feedback ? "bindless_heap_feedback" : "bindless_heap_translation");
    }
    return variable;
}

std::uint32_t LoadSampledImageDescriptor(SpirvEmitterState& state, std::uint32_t resource, std::uint32_t slotId) {
    const auto& imageResource = state.runtimeImage != nullptr ? *state.runtimeImage : state.program.Info().images.at(resource);
    if (imageResource.resourceClass != ImageResourceClass::Sampled) {
        FailEmit("sampled image descriptor requested for a non-sampled image");
    }
    // The heap branch of a table root (EmitImage): the entry comes from the heap set.
    // Experiment: APS5_HEAP_ROOT_IMAGE=1 samples the table root's own binding (the representative
    // entry) in the heap branch of graphics stages instead of the heap set.
    static const bool rootImage = std::getenv("APS5_HEAP_ROOT_IMAGE") != nullptr;
    if (state.bindlessHeapSlot != 0u && !(rootImage && state.program.Resources().stage != IrShaderStage::Compute) && state.program.Info().images.at(resource).indirectRoot == resource) {
        return LoadBindlessHeapImage(state, state.program.Info().images.at(resource), imageResource, state.bindlessHeapSlot);
    }
    const auto kind = DescriptorBindingForImage(imageResource);
    const auto arrayIndex = ResourceForDescriptor(state, kind, resource);
    const auto variable = state.imageVariables.at(ImageBindingIndex(kind));
    const auto imageType = ImageType(state, imageResource);
    const auto pointerType = state.module.Type(spv::OpTypePointer, spv::StorageClassUniformConstant, imageType);
    if (variable == 0) {
        ExitDescriptorBindingFailure(state, kind, resource, "sampled image descriptor array was not emitted");
    }
    std::uint32_t pointer = 0;
    if (slotId == 0) {
        pointer = DescriptorElementPointer(state, pointerType, variable, arrayIndex, kind, resource, "sampled image descriptor array was not emitted");
    } else {
        pointer = state.module.AllocateId();
        state.module.AddFunction(spv::OpAccessChain, pointerType, pointer, variable, TableImageIndex(state, kind, resource));
        DecorateTableNonUniform(state, slotId, pointer);
    }
    const auto image = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, imageType, image, pointer);
    DecorateTableNonUniform(state, slotId, image);
    return image;
}

std::uint32_t LoadSamplerDescriptor(SpirvEmitterState& state, std::uint32_t sampler) {
    const bool point = state.runtimeImage != nullptr && (state.runtimeImage->numericClass == IrTextureNumericClass::Sint || state.runtimeImage->conversionFormat != IrBufferFormat::Invalid || state.runtimeImage->depthBits);
    const auto arrayIndex = ResourceForDescriptor(state, DescriptorBindingKind::Samplers, sampler) + (point ? 1u : 0u);
    const auto samplerType = state.module.Type(spv::OpTypeSampler);
    const auto pointerType = state.module.Type(spv::OpTypePointer, spv::StorageClassUniformConstant, samplerType);
    const auto pointer = DescriptorElementPointer(state, pointerType, state.samplerVariable, arrayIndex, DescriptorBindingKind::Samplers, sampler, "sampler descriptor array was not emitted");
    const auto samplerId = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, samplerType, samplerId, pointer);
    return samplerId;
}

std::uint32_t MakeSampledImage(SpirvEmitterState& state, std::uint32_t resource, std::uint32_t sampler, std::uint32_t slotId) {
    const auto& imageResource = state.runtimeImage != nullptr ? *state.runtimeImage : state.program.Info().images.at(resource);
    const auto image = LoadSampledImageDescriptor(state, resource, slotId);
    const auto samplerId = LoadSamplerDescriptor(state, sampler);
    const auto sampledImage = state.module.AllocateId();
    const auto sampledType = state.module.Type(spv::OpTypeSampledImage, ImageType(state, imageResource));
    state.module.AddFunction(spv::OpSampledImage, sampledType, sampledImage, image, samplerId);
    DecorateTableNonUniform(state, slotId, sampledImage);
    if (state.bindlessHeapSlot != 0u && !state.bindlessHeapUniformSlot && state.program.Info().images.at(resource).indirectRoot == resource) state.module.AddAnnotation(spv::OpDecorate, sampledImage, spv::DecorationNonUniform);
    return sampledImage;
}

std::uint32_t StorageImageDescriptorPointer(SpirvEmitterState& state, std::uint32_t resource) {
    const auto& image = state.runtimeImage != nullptr ? *state.runtimeImage : state.program.Info().images.at(resource);
    if (image.resourceClass != ImageResourceClass::Storage) {
        FailEmit("storage image descriptor requested for a non-storage image");
    }
    const auto kind = DescriptorBindingForImage(image);
    const auto arrayIndex = ResourceForDescriptor(state, kind, resource);
    const auto pointerType = state.module.Type(spv::OpTypePointer, spv::StorageClassUniformConstant, ImageType(state, image));
    const auto variable = state.imageVariables.at(ImageBindingIndex(kind));
    return DescriptorElementPointer(state, pointerType, variable, arrayIndex, kind, resource, "storage image descriptor array was not emitted");
}

void EmitStorageImageWrite(SpirvEmitterState& state, std::uint32_t resource, std::uint32_t mipLod, std::uint32_t coord, std::uint32_t texel, std::uint32_t sample) {
    const auto& image = state.runtimeImage != nullptr ? *state.runtimeImage : state.program.Info().images.at(resource);
    if (image.resourceClass != ImageResourceClass::Storage) {
        FailEmit("storage image write requested for a non-storage image");
    }
    if (!image.atomic) {
        state.module.EmitCapability(spv::CapabilityStorageImageWriteWithoutFormat);
    }
    const auto kind = DescriptorBindingForImage(image);
    const auto arrayIndex = ResourceForDescriptor(state, kind, resource);
    const auto imageType = ImageType(state, image);
    const auto pointerType = state.module.Type(spv::OpTypePointer, spv::StorageClassUniformConstant, imageType);
    const auto variable = state.imageVariables.at(ImageBindingIndex(kind));
    const auto loadAt = [&](std::uint32_t index) {
        const auto pointer = DescriptorElementPointer(state, pointerType, variable, index, kind, resource, "storage image descriptor array was not emitted");
        const auto descriptor = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, imageType, descriptor, pointer);
        return descriptor;
    };
    if (RdnaImageDimensionInfoFor(image.dimension).multisampled != 0u) {
        if (sample == 0u || image.mipMode == ImageMipMode::DynamicStorage) FailEmit("multisampled storage image requires a sample and cannot have mip levels");
        state.module.AddFunction(spv::OpImageWrite, loadAt(arrayIndex), coord, texel, spv::ImageOperandsSampleMask, sample);
        return;
    }
    if (image.mipMode != ImageMipMode::DynamicStorage) {
        state.module.AddFunction(spv::OpImageWrite, loadAt(arrayIndex), coord, texel);
        return;
    }
    if (image.mipCount == 0u) {
        ExitDescriptorBindingFailure(state, kind, resource, "dynamic storage image has no mip descriptors");
    }
    const auto count = state.registrationModules ? RuntimeMipCount(state, resource) : state.module.SpecializationConstant(TypeU32(state), PipelineSpecialization::MipCountBase + resource, image.mipCount);
    const auto mergeLabel = state.module.AllocateId();
    std::vector<std::uint32_t> labels(image.mipCount);
    std::vector<std::uint32_t> words{spv::OpSwitch, mipLod, mergeLabel};
    for (std::uint32_t mip = 0; mip < image.mipCount; mip++) {
        labels[mip] = state.module.AllocateId();
        words.push_back(mip);
        words.push_back(labels[mip]);
    }
    state.module.AddFunction(spv::OpSelectionMerge, mergeLabel, spv::SelectionControlMaskNone);
    state.module.AddFunction(std::span<const std::uint32_t>(words));
    for (std::uint32_t mip = 0; mip < image.mipCount; mip++) {
        EmitLabel(state, labels[mip]);
        const auto active = Binary(state, spv::OpULessThan, TypeBool(state), ConstantU32(state, mip), count);
        const auto writeLabel = state.module.AllocateId();
        const auto nextLabel = state.module.AllocateId();
        state.module.AddFunction(spv::OpSelectionMerge, nextLabel, spv::SelectionControlMaskNone);
        state.module.AddFunction(spv::OpBranchConditional, active, writeLabel, nextLabel);
        EmitLabel(state, writeLabel);
        state.module.AddFunction(spv::OpImageWrite, loadAt(arrayIndex + mip), coord, texel);
        state.module.AddFunction(spv::OpBranch, nextLabel);
        EmitLabel(state, nextLabel);
        state.module.AddFunction(spv::OpBranch, mergeLabel);
    }
    EmitLabel(state, mergeLabel);
}

const RdnaImageDimensionInfo& RdnaImageDimensionInfoFor(RdnaImageDimension dimension) {
    for (const auto& info : ImageDimensions) {
        if (info.dimension == dimension) {
            return info;
        }
    }
    FailEmit("image dimension " + std::to_string(static_cast<std::uint32_t>(dimension)) + " is invalid");
}

}
