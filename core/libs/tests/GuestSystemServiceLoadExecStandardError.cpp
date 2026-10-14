#include "prx/libc/include/general/VabiMacros.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

extern "C" {
int APS5_VABI sceSystemServiceLoadExec(const char*, const char* const*);
int APS5_VABI dup2_nid_postfix(int, int);
}

static void Require(bool value) { if (!value) std::abort(); }

static std::string Drain(int descriptor) {
    std::string text;
    char buffer[4096];
    for (ssize_t count; (count = ::read(descriptor, buffer, sizeof(buffer))) > 0;) text.append(buffer, static_cast<std::size_t>(count));
    ::close(descriptor);
    return text;
}

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--after-load-exec") == 0) {
        std::fputs("started\n", stderr);
        return 0;
    }
    Require(argc == 1);
    int host[2] = {-1, -1};
    int guest[2] = {-1, -1};
    Require(::pipe(host) == 0);
    Require(::pipe(guest) == 0);
    const pid_t child = ::fork();
    Require(child >= 0);
    if (child == 0) {
        ::close(host[0]);
        ::close(guest[0]);
        if (::dup2(host[1], 2) != 2) ::_exit(70);
        ::close(host[1]);
        if (dup2_nid_postfix(guest[1], 2) != 2) ::_exit(71);
        ::close(guest[1]);
        const char* const arguments[] = {"--after-load-exec", nullptr};
        sceSystemServiceLoadExec("/app0/eboot.bin", arguments);
        ::_exit(72);
    }
    ::close(host[1]);
    ::close(guest[1]);
    const std::string hostText = Drain(host[0]);
    const std::string guestText = Drain(guest[0]);
    int status = 0;
    Require(::waitpid(child, &status, 0) == child);
    Require(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    Require(hostText == "started\n");
    Require(guestText.empty());
    return 0;
}
