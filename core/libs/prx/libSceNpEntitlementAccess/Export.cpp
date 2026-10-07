#include <cstdint>
#include <cstddef>
#include "SceTypes.hpp"
#include "prx/libc/include/General.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

static constexpr int SCE_NP_ENTITLEMENT_ACCESS_ERROR_PARAMETER = static_cast<int>(0x80558003);
static constexpr int SCE_NP_ENTITLEMENT_ACCESS_ERROR_NOT_FOUND = static_cast<int>(0x80558007);
static constexpr int SCE_NP_ERROR_SIGNED_OUT = static_cast<int>(0x80550006);
static constexpr uint32_t SKU_FLAG_FULL = 3;

namespace {

std::filesystem::path EntitlementsPath() {
    if (const char* configured = std::getenv("ANYPS5_ENTITLEMENTS"); configured != nullptr && configured[0] != '\0') return configured;
#ifdef _WIN32
    wchar_t module[MAX_PATH];
    const auto length = GetModuleFileNameW(nullptr, module, MAX_PATH);
    if (length == 0 || length == MAX_PATH) throw std::runtime_error("NpEntitlementAccess: cannot locate the executable");
    return std::filesystem::path(module).parent_path() / "anyps5-entitlements.ini";
#elif defined(__APPLE__)
    std::uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::vector<char> executable(size);
    std::error_code error;
    std::filesystem::path path;
    if (_NSGetExecutablePath(executable.data(), &size) == 0) path = std::filesystem::canonical(executable.data(), error);
    if (path.empty()) throw std::runtime_error("NpEntitlementAccess: cannot locate the executable");
    return path.parent_path() / "anyps5-entitlements.ini";
#else
    return std::filesystem::read_symlink("/proc/self/exe").parent_path() / "anyps5-entitlements.ini";
#endif
}

const std::vector<NpEntitlementAccessAddcontEntitlementInfo>& OwnedAddons() {
    static const auto owned = [] {
        std::vector<NpEntitlementAccessAddcontEntitlementInfo> addons;
        const auto path = EntitlementsPath();
        std::ifstream file(path);
        if (!file) {
            if (std::getenv("ANYPS5_ENTITLEMENTS") != nullptr || std::filesystem::exists(path)) throw std::runtime_error("NpEntitlementAccess: cannot read " + path.string());
            return addons;
        }
        std::string line;
        while (std::getline(file, line)) {
            line.erase(std::min(line.find_first_of("#;"), line.size()));
            const auto first = line.find_first_not_of(" \t\r");
            if (first == std::string::npos) continue;
            const auto label = line.substr(first, line.find_last_not_of(" \t\r") + 1 - first);
            NpEntitlementAccessAddcontEntitlementInfo info{};
            if (label.size() >= sizeof(info.entitlement_label.data)) throw std::runtime_error("NpEntitlementAccess: entitlement label '" + label + "' in " + path.string() + " is longer than 16 characters");
            std::memcpy(info.entitlement_label.data, label.data(), label.size());
            info.package_type = 3;
            info.download_status = 4;
            addons.push_back(info);
        }
        return addons;
    }();
    return owned;
}

}

extern "C" {

int APS5_VABI sceNpEntitlementAccessGetAddcontEntitlementInfo(uint32_t service_label, const NpUnifiedEntitlementLabel* entitlement_label, NpEntitlementAccessAddcontEntitlementInfo* info) {
    (void)service_label;
    if (!entitlement_label || !info) return SCE_NP_ENTITLEMENT_ACCESS_ERROR_PARAMETER;
    for (const auto& addon : OwnedAddons()) {
        if (std::strncmp(addon.entitlement_label.data, entitlement_label->data, sizeof(entitlement_label->data)) == 0) {
            *info = addon;
            return 0;
        }
    }
    return SCE_NP_ENTITLEMENT_ACCESS_ERROR_NOT_FOUND;
}

int APS5_VABI sceNpEntitlementAccessGetAddcontEntitlementInfoList(uint32_t service_label, NpEntitlementAccessAddcontEntitlementInfo* list, uint32_t list_num, uint32_t* hit_num) {
    (void)service_label;
    if (!hit_num || (!list && list_num != 0)) return SCE_NP_ENTITLEMENT_ACCESS_ERROR_PARAMETER;
    const auto& owned = OwnedAddons();
    *hit_num = static_cast<uint32_t>(owned.size());
    std::copy_n(owned.begin(), std::min<std::size_t>(list_num, owned.size()), list);
    return 0;
}

int APS5_VABI sceNpEntitlementAccessGetSkuFlag(uint32_t* sku_flag) {
    if (!sku_flag) return SCE_NP_ENTITLEMENT_ACCESS_ERROR_PARAMETER;
    *sku_flag = SKU_FLAG_FULL;
    return 0;
}

int APS5_VABI sceNpEntitlementAccessGetEntitlementKey(
    uint32_t serviceLabel, const NpUnifiedEntitlementLabel* entitlementLabel,
    NpEntitlementAccessEntitlementKey* key) {
    (void)serviceLabel;
    constexpr int errorParameter = static_cast<int>(0x817D0002);
    constexpr int errorNoEntitlement = static_cast<int>(0x817D0007);
    if (!entitlementLabel || !key) return errorParameter;
    return errorNoEntitlement;
}

int APS5_VABI sceNpEntitlementAccessInitialize(const NpEntitlementAccessInitParam* init_param, NpEntitlementAccessBootParam* boot_param) {
    (void)init_param;
    (void)boot_param;
    return 0;
}

// There is no entitlement server: a list request has finished when it is created, and the console
// owns no unified or service entitlements (consumables, currency, subscriptions), so every list
// polls empty with both paging offsets invalid (add-ons come from GetAddcontEntitlementInfoList).
static constexpr int SCE_NP_ENTITLEMENT_ACCESS_ERROR_REQUEST_NOT_FOUND = static_cast<int>(0x817D0015);
static constexpr int SCE_NP_ENTITLEMENT_ACCESS_ERROR_ABORTED = static_cast<int>(0x817D0016);
static constexpr int64_t REQUEST_ID_OFFSET = 0x10000000;
static constexpr int32_t INVALID_OFFSET = -1;

static std::mutex g_requestMutex;
static int64_t g_requestCount = 0;
static std::map<int64_t, bool> g_requests; // id -> aborted

static int CreateListRequest(const void* param, const void* list, uint32_t list_num, int64_t* request_id) {
    if (!param || !request_id || (!list && list_num != 0)) return SCE_NP_ENTITLEMENT_ACCESS_ERROR_PARAMETER;
    std::lock_guard lock(g_requestMutex);
    const auto id = REQUEST_ID_OFFSET + ++g_requestCount;
    g_requests[id] = false;
    *request_id = id;
    return 0;
}

static int PollEmptyList(int64_t request_id, int* result, uint32_t* hit_num, int32_t* next_offset, int32_t* previous_offset) {
    if (!result) return SCE_NP_ENTITLEMENT_ACCESS_ERROR_PARAMETER;
    std::lock_guard lock(g_requestMutex);
    const auto request = g_requests.find(request_id);
    if (request == g_requests.end()) return SCE_NP_ENTITLEMENT_ACCESS_ERROR_REQUEST_NOT_FOUND;
    *result = request->second ? SCE_NP_ENTITLEMENT_ACCESS_ERROR_ABORTED : 0;
    if (hit_num) *hit_num = 0;
    if (next_offset) *next_offset = INVALID_OFFSET;
    if (previous_offset) *previous_offset = INVALID_OFFSET;
    return 0;
}

int APS5_VABI sceNpEntitlementAccessAbortRequest(int64_t request_id) {
    std::lock_guard lock(g_requestMutex);
    const auto request = g_requests.find(request_id);
    if (request == g_requests.end()) return SCE_NP_ENTITLEMENT_ACCESS_ERROR_REQUEST_NOT_FOUND;
    request->second = true;
    return 0;
}

int APS5_VABI sceNpEntitlementAccessDeleteRequest(int64_t request_id) {
    std::lock_guard lock(g_requestMutex);
    return g_requests.erase(request_id) == 0 ? SCE_NP_ENTITLEMENT_ACCESS_ERROR_REQUEST_NOT_FOUND : 0;
}

int APS5_VABI sceNpEntitlementAccessGenerateTransactionId(void* transaction_id) {
 if (!transaction_id) return SCE_NP_ENTITLEMENT_ACCESS_ERROR_PARAMETER;
 return SCE_NP_ERROR_SIGNED_OUT;
}

int APS5_VABI sceNpEntitlementAccessPollConsumeEntitlement(void) {
 return SCE_NP_ERROR_SIGNED_OUT;
}

int APS5_VABI sceNpEntitlementAccessRequestConsumeUnifiedEntitlement(void) {
 return SCE_NP_ERROR_SIGNED_OUT;
}


int APS5_VABI sceNpEntitlementAccessRequestConsumeServiceEntitlement(void) {
 return SCE_NP_ERROR_SIGNED_OUT;
}


int APS5_VABI sceNpEntitlementAccessPollServiceEntitlementInfo(void) {
    return SCE_NP_ERROR_SIGNED_OUT;
}

int APS5_VABI sceNpEntitlementAccessPollUnifiedEntitlementInfo(void) {
    return SCE_NP_ERROR_SIGNED_OUT;
}

int APS5_VABI sceNpEntitlementAccessRequestServiceEntitlementInfo(void) {
    return SCE_NP_ERROR_SIGNED_OUT;
}

int APS5_VABI sceNpEntitlementAccessRequestUnifiedEntitlementInfo(void) {
    return SCE_NP_ERROR_SIGNED_OUT;
}

int APS5_VABI sceNpEntitlementAccessGetPftFlag(uint32_t* flag) {
    constexpr int errorParameter = static_cast<int>(0x817D0002);
    if (!flag) return errorParameter;
    *flag = 0;
    return 0;
}

int APS5_VABI sceNpEntitlementAccessGetGameTrialsFlag(uint32_t* flag) {
    constexpr int errorParameter = static_cast<int>(0x817D0002);
    if (!flag) return errorParameter;
    *flag = 0;
    return 0;
}

int APS5_VABI sceNpEntitlementAccessPollServiceEntitlementInfoList(int64_t request_id, int* result, void* list, uint32_t list_num, uint32_t* hit_num, int32_t* next_offset, int32_t* previous_offset) {
    (void)list;
    (void)list_num;
    return PollEmptyList(request_id, result, hit_num, next_offset, previous_offset);
}

int APS5_VABI sceNpEntitlementAccessPollUnifiedEntitlementInfoList(int64_t request_id, int* result, void* list, uint32_t list_num, uint32_t* hit_num, int32_t* next_offset, int32_t* previous_offset) {
    (void)list;
    (void)list_num;
    return PollEmptyList(request_id, result, hit_num, next_offset, previous_offset);
}

int APS5_VABI sceNpEntitlementAccessRequestServiceEntitlementInfoList(int user_id, uint32_t service_label, const void* param, int64_t* request_id) {
    (void)user_id;
    (void)service_label;
    return CreateListRequest(param, nullptr, 0, request_id);
}

int APS5_VABI sceNpEntitlementAccessRequestUnifiedEntitlementInfoList(int user_id, uint32_t service_label, const NpUnifiedEntitlementLabel* list, uint32_t list_num, const void* param, int64_t* request_id) {
    (void)user_id;
    (void)service_label;
    return CreateListRequest(param, list, list_num, request_id);
}

}
