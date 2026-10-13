#include "prx/libc/include/general/VabiMacros.hpp"
#include <cstddef>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <string>

extern "C" {
int APS5_VABI sceSslInit_nid_postfix(std::size_t);
int APS5_VABI sceSslTerm_nid_postfix(int);
int APS5_VABI sceSslGetCaCerts(int, void*);
int APS5_VABI sceSslFreeCaCerts(int, void*);
int APS5_VABI sceSslGetCaList(int, void*);
int APS5_VABI sceSslFreeCaList(int, void*);
int APS5_VABI sceSslUnloadCert(int);
int APS5_VABI sceSslCreateConnection(int, int, const char*);
int APS5_VABI sceSslDeleteConnection(int);
int APS5_VABI sceSslSetVerifyCallback(int, int (APS5_VABI *)(int, unsigned int, void**, int, void*), void*);
int APS5_VABI sceSslConnect(int);
int APS5_VABI sceSslRead(int, void*, std::size_t, int);
int APS5_VABI sceSslWrite(int, void*, std::size_t, int);
int APS5_VABI sceNetSocket(const char*, int, int, int);
int APS5_VABI sceNetBind_nid_postfix(int, const void*, std::uint32_t);
int APS5_VABI sceNetListen(int, int);
int APS5_VABI sceNetGetsockname(int, void*, std::uint32_t*);
int APS5_VABI sceNetConnect(int, const void*, std::uint32_t);
int APS5_VABI sceNetSocketClose(int);
}

struct SslMemoryPoolStats {
    std::size_t pool_size;
    std::size_t max_inuse_size;
    std::size_t current_inuse_size;
    std::int32_t reserved;
};

extern "C" int APS5_VABI sceSslGetMemoryPoolStats(int, SslMemoryPoolStats*);

static void Require(bool value) { if (!value) std::abort(); }

template <typename TCall>
static bool Throws(TCall call) {
    try {
        call();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

template <typename TCall>
static bool ThrowsWith(TCall call, const std::string& text) {
    try {
        call();
    } catch (const std::exception& error) {
        return std::string(error.what()).find(text) != std::string::npos;
    }
    return false;
}

static int APS5_VABI VerifyCallback(int, unsigned int, void**, int, void*) { return 0; }

struct SslCaCerts {
    void* certs;
    std::size_t num;
    void* pool;
};

struct SslCaList {
    void** certs;
    int num;
};

int main() {
    constexpr int notFound = static_cast<int>(0x8095F004);
    constexpr int invalidArg = static_cast<int>(0x8095177A);
    int marker = 0;
    const int context = sceSslInit_nid_postfix(0x10000);
    Require(context > 0);
    SslMemoryPoolStats stats{1, 1, 1, 1};
    Require(sceSslGetMemoryPoolStats(context, &stats) == 0);
    Require(stats.pool_size == 0x10000 && stats.max_inuse_size == 0 && stats.current_inuse_size == 0 && stats.reserved == 0);
    Require(sceSslGetCaCerts(context, nullptr) == invalidArg);
    Require(sceSslFreeCaCerts(context, nullptr) == invalidArg);

    SslCaCerts certs{&marker, 3, &marker};
    Require(sceSslGetCaCerts(context, &certs) == notFound);
    Require(certs.certs == nullptr && certs.num == 0 && certs.pool == nullptr);

    certs = {&marker, 3, &marker};
    Require(sceSslFreeCaCerts(context, &certs) == 0);
    Require(certs.certs == nullptr && certs.num == 0 && certs.pool == nullptr);

    Require(sceSslGetCaList(context, nullptr) == invalidArg);
    Require(sceSslFreeCaList(context, nullptr) == invalidArg);

    void* handle = &marker;
    SslCaList list{&handle, 3};
    Require(sceSslGetCaList(context, &list) == notFound);
    Require(list.certs == nullptr && list.num == 0);
    Require(sceSslFreeCaList(context, &list) == 0);
    Require(list.certs == nullptr && list.num == 0);

    SslCaList foreign{&handle, 1};
    Require(Throws([&] { sceSslFreeCaList(context, &foreign); }));
    foreign = {nullptr, 1};
    Require(Throws([&] { sceSslFreeCaList(context, &foreign); }));
    foreign = {&handle, 0};
    Require(Throws([&] { sceSslFreeCaList(context, &foreign); }));

    Require(sceSslUnloadCert(context) == 0);
    Require(sceSslGetCaList(context, &list) == notFound);
    Require(list.certs == nullptr && list.num == 0);

    constexpr int invalidId = static_cast<int>(0x8095F006);
    Require(Throws([&] { sceSslCreateConnection(context, 0, nullptr); }));
    Require(sceSslCreateConnection(context + 1000, 0, "example.com") == invalidId);

    const int listener = sceNetSocket(nullptr, 2, 1, 6);
    Require(listener >= 0);
    std::array<std::uint8_t, 16> address{16, 2, 0, 0, 127, 0, 0, 1};
    Require(sceNetBind_nid_postfix(listener, address.data(), address.size()) == 0);
    Require(sceNetListen(listener, 1) == 0);
    std::uint32_t addressSize = address.size();
    Require(sceNetGetsockname(listener, address.data(), &addressSize) == 0);
    const std::string endpoint = "example.com:" + std::to_string((address[2] << 8) | address[3]);
    const int client = sceNetSocket(nullptr, 2, 1, 6);
    Require(client >= 0 && sceNetConnect(client, address.data(), address.size()) == 0);
    const int idle = sceNetSocket(nullptr, 2, 1, 6);
    Require(idle >= 0);

    const int connection = sceSslCreateConnection(context, client, "example.com");
    Require(connection > 0 && connection != context);
    Require(sceSslSetVerifyCallback(connection, VerifyCallback, &marker) == 0);
    Require(sceSslSetVerifyCallback(connection, nullptr, nullptr) == 0);
    char buffer[4]{};
    Require(ThrowsWith([&] { sceSslConnect(connection); }, endpoint));
    Require(ThrowsWith([&] { sceSslRead(connection, buffer, sizeof(buffer), 0); }, endpoint));
    Require(ThrowsWith([&] { sceSslWrite(connection, buffer, sizeof(buffer), 0); }, endpoint));

    const int unconnected = sceSslCreateConnection(context, idle, "example.com");
    Require(unconnected > 0 && unconnected != connection);
    Require(ThrowsWith([&] { sceSslConnect(unconnected); }, "example.com (socket"));
    Require(sceSslDeleteConnection(unconnected) == 0);

    const int unknown = connection + 1000;
    Require(Throws([&] { sceSslSetVerifyCallback(unknown, nullptr, nullptr); }));
    Require(ThrowsWith([&] { sceSslConnect(unknown); }, "unknown connection"));
    Require(ThrowsWith([&] { sceSslRead(unknown, buffer, sizeof(buffer), 0); }, "unknown connection"));
    Require(ThrowsWith([&] { sceSslWrite(unknown, buffer, sizeof(buffer), 0); }, "unknown connection"));
    Require(Throws([&] { sceSslDeleteConnection(unknown); }));
    Require(Throws([&] { sceSslConnect(unconnected); }));

    Require(sceSslTerm_nid_postfix(context) == 0);
    Require(Throws([&] { sceSslSetVerifyCallback(connection, nullptr, nullptr); }));
    Require(ThrowsWith([&] { sceSslConnect(connection); }, "terminated"));
    Require(sceSslCreateConnection(context, client, "example.com") == invalidId);
    Require(sceSslDeleteConnection(connection) == 0);
    Require(Throws([&] { sceSslDeleteConnection(connection); }));
    Require(sceNetSocketClose(idle) == 0);
    Require(sceNetSocketClose(client) == 0);
    Require(sceNetSocketClose(listener) == 0);

    Require(sceSslGetCaList(context, nullptr) == invalidArg);
    Require(sceSslFreeCaList(context, nullptr) == invalidArg);
    list = {&handle, 3};
    Require(Throws([&] { sceSslGetCaList(context, &list); }));
    Require(Throws([&] { sceSslFreeCaList(context, &list); }));
    Require(Throws([&] { sceSslUnloadCert(context); }));
}
