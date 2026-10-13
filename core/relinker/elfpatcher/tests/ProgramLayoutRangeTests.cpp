#include <elfpatcher/general/ProgramHeaderLayoutBuilder.hpp>
#include <io/ByteWriter.hpp>
#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>

class KeepAll : public Elfpatcher::ISegmentFilter {
public:
    bool ShouldSkip(const Domain::ProgramHeader&) const override { return false; }
};

static int gLine=0;
static void Require(bool ok) { if (!ok) throw std::runtime_error("Program layout range validation failed at line " + std::to_string(gLine)); }

static Domain::ProgramHeader Load(std::uint64_t vaddr, std::uint64_t memorySize, std::uint64_t fileSize, std::uint32_t flags) {
    Domain::ProgramHeader load{};
    load.Type = 1; load.Flags = flags; load.Offset = vaddr; load.MappedAddress = vaddr;
    load.PhysicalAddress = vaddr; load.FileSize = fileSize; load.MemorySize = memorySize; load.Alignment = 0x1000;
    return load;
}

static std::vector<Domain::ProgramHeader> MappedLoads(const std::vector<std::uint8_t>& buf, std::uint64_t tableOffset, std::uint16_t count) {
    std::vector<Domain::ProgramHeader> loads;
    for (std::uint16_t index = 0; index < count; ++index) {
        const auto* entry = buf.data() + tableOffset + index * 56;
        if (*reinterpret_cast<const std::uint32_t*>(entry) != 1)
            continue;
        Domain::ProgramHeader ph{};
        ph.Flags = *reinterpret_cast<const std::uint32_t*>(entry + 4);
        ph.Offset = *reinterpret_cast<const std::uint64_t*>(entry + 8);
        ph.MappedAddress = *reinterpret_cast<const std::uint64_t*>(entry + 16);
        ph.FileSize = *reinterpret_cast<const std::uint64_t*>(entry + 32);
        ph.MemorySize = *reinterpret_cast<const std::uint64_t*>(entry + 40);
        ph.Alignment = *reinterpret_cast<const std::uint64_t*>(entry + 48);
        loads.push_back(ph);
    }
    return loads;
}

static void RequireExtraBlockStartsOnItsOwnPage(std::vector<Domain::ProgramHeader> loads, std::uint64_t extraFileOffset) {
    gLine = 3000;
    std::sort(loads.begin(), loads.end(), [](const Domain::ProgramHeader& left, const Domain::ProgramHeader& right) {
        return left.MappedAddress < right.MappedAddress;
    });
    std::size_t extraIndex = loads.size();
    for (std::size_t index = 0; index < loads.size(); ++index) {
        if (loads[index].FileSize > loads[index].MemorySize) std::cerr << "case3005 off=" << std::hex << loads[index].Offset << " vaddr=" << loads[index].MappedAddress << " fsz=" << loads[index].FileSize << " msz=" << loads[index].MemorySize << "\n";
        gLine = 3005; Require(loads[index].FileSize <= loads[index].MemorySize);
        gLine = 3006; Require(loads[index].MappedAddress % loads[index].Alignment == loads[index].Offset % loads[index].Alignment);
        if (loads[index].Offset == extraFileOffset)
            extraIndex = index;
    }
    gLine = 3001; Require(extraIndex < loads.size());
    const auto& extraBlock = loads[extraIndex];
    gLine = 3002; Require(extraBlock.MemorySize >= extraBlock.FileSize);
    for (std::size_t index = 0; index < loads.size(); ++index) {
        if (index == extraIndex)
            continue;
        const auto& other = loads[index];
        if (other.MappedAddress < extraBlock.MappedAddress) {
            gLine = 3003; Require((other.MappedAddress + other.FileSize - 1) / 0x1000 < extraBlock.MappedAddress / 0x1000);
        } else {
            gLine = 3004; Require(other.MappedAddress >= extraBlock.MappedAddress + extraBlock.FileSize);
        }
    }
}

int main() {
    try {
        auto byteWriter = std::make_shared<Io::ByteWriter>();
        Elfpatcher::ProgramHeaderLayoutBuilder builder(std::make_shared<KeepAll>(), byteWriter);
        constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
        Domain::ProgramHeader load{};
        load.Type = 1; load.MappedAddress = maximum - 3; load.MemorySize = 8;
        bool rejected = false;
        try { builder.ComputeExtraBlockVaddr({load}, 0x1000); } catch (const std::exception&) { rejected = true; }
        gLine=__LINE__; Require(rejected);
        load.MappedAddress = 0x1000; load.MemorySize = 0x1000;
        gLine=__LINE__; Require(builder.ComputeExtraBlockVaddr({load}, 0x1003) == 0x2003);
        load.MemorySize = 0x1800;
        gLine=__LINE__; Require(builder.ComputeExtraBlockVaddr({load}, 0x8A00) == 0x3A00);
        gLine=__LINE__; Require(builder.ComputeExtraBlockVaddr({load}, 0x8800) == 0x3800);
        load.MappedAddress = 0x1000; load.MemorySize = 0x1000;
        gLine=__LINE__; Require(builder.ComputeExtraBlockVaddr({load}, 0x1000) == 0x2000);
        Elfpatcher::ProgramHeaderLayoutRequest valid{64, 56, 6, {load}, 0x1000, 0x2000, 0x100, 0x1000, 16, 0x1010, 8};
        std::vector<std::uint8_t> output(0x1100, 0xA5);
        gLine=__LINE__; Require(builder.WriteLayout(output, valid) == 6);
        for (int kind = 0; kind < 11; ++kind) {
            auto request = valid;
            switch (kind) {
                case 0: request.PhEntSize = 55; break;
                case 1: request.PhOff = maximum - 10; break;
                case 2: request.PhOff = 0x10F0; break;
                case 3: request.ExtraBlockVaddr = maximum - 10; break;
                case 4: request.DynamicSegmentSize = maximum; break;
                case 5: request.InterpSize = maximum; break;
                case 6: request.DynamicSegmentOffset = maximum; break;
                case 7: request.InterpOffset = maximum; break;
                case 8: request.ExtraBlockOffset = maximum; break;
                case 9: request.ExtraBlockSize = 0x1000; break;
                case 10: request.PhOff = 0xF00; request.ExtraBlockVaddr = maximum - 0x1fff; break;
            }
            std::vector<std::uint8_t> bytes(0x1100, 0xA5);
            const auto before = bytes;
            rejected = false;
            try { builder.WriteLayout(bytes, request); } catch (const std::exception&) { rejected = true; }
            gLine=__LINE__; Require(rejected && bytes == before);
        }
        const std::uint16_t tableCount = 2 + 5;
        const auto writeLayout = [&builder, tableCount](const std::vector<Domain::ProgramHeader>& headers, std::uint64_t extraFileOffset, std::uint64_t extraFileSize) {
            const std::uint64_t tableSize = tableCount * 56;
            std::vector<std::uint8_t> bytes(std::max<std::uint64_t>(0xB000, extraFileOffset + tableSize + extraFileSize), 0);
            Elfpatcher::ProgramHeaderLayoutRequest request{extraFileOffset, 56, tableCount, headers, extraFileOffset,
                builder.ComputeExtraBlockVaddr(headers, extraFileOffset), tableSize + extraFileSize,
                extraFileOffset, 0x20, extraFileOffset + 0x20, 8};
            const std::uint16_t written = builder.WriteLayout(bytes, request);
            gLine=__LINE__; Require(written == tableCount);
            RequireExtraBlockStartsOnItsOwnPage(MappedLoads(bytes, extraFileOffset, written), extraFileOffset);
        };
        writeLayout({Load(0x1000, 0x1800, 0x1800, 6), Load(0x3000, 0x1000, 0x1000, 4)}, 0x8A00, 0x400);
        writeLayout({Load(0x1000, 0x8000, 0x800, 6), Load(0xA000, 0x1000, 0x1000, 4)}, 0x8A00, 0x400);
        writeLayout({Load(0x1000, 0x8000, 0x800, 6), Load(0xA000, 0x1000, 0x1000, 4)}, 0x8000, 0x400);
        writeLayout({Load(0x1000, 0x2000, 0x800, 6), Load(0x3000, 0x3000, 0x1000, 4)}, 0x8A00, 0x400);
        writeLayout({Load(0x1000, 0x1200, 0x1200, 5), Load(0x2400, 0x1000, 0x1000, 6)}, 0x8A00, 0x400);
        writeLayout({Load(0x1000, 0x3000, 0x1000, 5), Load(0x3000, 0x4000, 0x1000, 6)}, 0x8A00, 0x400);
        writeLayout({Load(0x1000, 0x1000, 0x1000, 5), Load(0x2000, 0x1000, 0x0400, 6)}, 0x8800, 0x400);
        std::cout << "21 program layout cases passed\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}