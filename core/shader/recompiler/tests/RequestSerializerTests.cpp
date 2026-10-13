#include "ControlFlow/RequestSerializer.hpp"

#include <cstdlib>
#include <stdexcept>

namespace {

constexpr char kHistoricalVersion21[] =
    "NVNQQRUAAAAFAAADAAAAAAABAAAAAAAAAAAAgb8AAAAAAAAAAAAAAAAAAAAAQAAAAAgAAAADAAAAAAAAAN+bVxPgrGgk776t3gABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJAAAAAAAAAOQAAAAAAAAAAAAAAAAAAAAAAQEAAAEAAAAAAAAAAAAFAAAAAAAEAAAAAAAAABI0VngAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAAALAAAAEAAAAEAAAAAAAAF4VjQSAQEBOTAAAAA=";

void Require(bool condition) {
    if (!condition) std::abort();
}

}

int main() {
    const ShaderRecompiler::RequestSerializer serializer;

    const auto historical = serializer.Deserialize(kHistoricalVersion21);
    Require(!historical.request.target.float16DenormPreserve);

    ShaderRecompiler::RecompileRequest request{};
    request.target.float16DenormPreserve = false;
    const auto withoutPreserve = serializer.Deserialize(serializer.Serialize(request));
    Require(!withoutPreserve.request.target.float16DenormPreserve);

    request.target.float16DenormPreserve = true;
    const auto withPreserve = serializer.Deserialize(serializer.Serialize(request));
    Require(withPreserve.request.target.float16DenormPreserve);

    return 0;
}
