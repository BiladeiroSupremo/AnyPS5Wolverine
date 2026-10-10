#include "SceTypes.hpp"

#include <cstdint>
#include <cstdlib>

extern "C" {
int APS5_VABI sceNpCommerceDialogInitialize();
int APS5_VABI sceNpCommerceDialogOpen(const void* param);
int APS5_VABI sceNpCommerceDialogUpdateStatus(void);
int APS5_VABI sceNpCommerceDialogGetStatus(void);
int APS5_VABI sceNpCommerceDialogGetResult(void* result);
int APS5_VABI sceNpCommerceDialogClose(void);
int APS5_VABI sceNpCommerceDialogTerminate();
int APS5_VABI sceNpCommerceHidePsStoreIcon(void);
int APS5_VABI sceNpCommerceShowPsStoreIcon(int pos);
int APS5_VABI sceNpCommerceSetPsStoreIconLayout(int layout);
int APS5_VABI sceNpCommerceDialogOpen2(const void* param);
}

static void Require(bool condition) {
    if (!condition) {
        std::abort();
    }
}

int main() {
    constexpr int statusNone = 0;
    constexpr int statusInitialized = 1;
    constexpr int statusFinished = 3;
    constexpr int resultUserCanceled = 1;
    constexpr int errorNotInitialized = static_cast<int>(0x80B80003u);
    constexpr int errorAlreadyInitialized = static_cast<int>(0x80B80004u);
    constexpr int errorNotFinished = static_cast<int>(0x80B80006u);
    constexpr int errorArgNull = static_cast<int>(0x80B80009u);

    struct DialogResult {
        std::int32_t result;
        std::uint8_t authorized;
    };

    Require(sceNpCommerceDialogGetStatus() == statusNone);
    Require(sceNpCommerceDialogUpdateStatus() == statusNone);
    Require(sceNpCommerceDialogOpen(nullptr) == errorNotInitialized);
    Require(sceNpCommerceDialogOpen2(nullptr) == errorNotInitialized);
    Require(sceNpCommerceDialogClose() == errorNotInitialized);

    DialogResult dialogResult{};
    Require(sceNpCommerceDialogGetResult(&dialogResult) == errorNotFinished);

    Require(sceNpCommerceDialogInitialize() == 0);
    Require(sceNpCommerceDialogInitialize() == errorAlreadyInitialized);
    Require(sceNpCommerceDialogGetStatus() == statusInitialized);

    Require(sceNpCommerceDialogOpen(nullptr) == errorArgNull);
    Require(sceNpCommerceDialogOpen2(nullptr) == 0);
    Require(sceNpCommerceDialogGetStatus() == statusFinished);
    Require(sceNpCommerceDialogUpdateStatus() == statusFinished);

    Require(sceNpCommerceDialogGetResult(nullptr) == errorArgNull);
    Require(sceNpCommerceDialogGetResult(&dialogResult) == 0);
    Require(dialogResult.result == resultUserCanceled);
    Require(dialogResult.authorized == 0);

    Require(sceNpCommerceDialogClose() == 0);
    Require(sceNpCommerceDialogTerminate() == 0);
    Require(sceNpCommerceDialogGetStatus() == statusNone);

    alignas(8) char param[16] = {};
    Require(sceNpCommerceDialogInitialize() == 0);
    Require(sceNpCommerceDialogOpen2(param) == 0);
    Require(sceNpCommerceDialogGetStatus() == statusFinished);
    Require(sceNpCommerceDialogTerminate() == 0);

    Require(sceNpCommerceHidePsStoreIcon() == 0);
    Require(sceNpCommerceShowPsStoreIcon(0) == 0);
    Require(sceNpCommerceSetPsStoreIconLayout(0) == 0);

    return 0;
}
