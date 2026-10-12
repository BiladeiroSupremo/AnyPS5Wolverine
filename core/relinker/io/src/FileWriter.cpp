#include <io/FileWriter.hpp>
#include <io/NativePath.hpp>
#include <domain/Types.hpp>
#include <filesystem>
#include <fstream>
#include <system_error>

namespace Io {

namespace {

template<typename TFill>
bool writeFile(const std::filesystem::path& file, const std::string& path, const std::ios::openmode mode, const TFill& fill) {
    std::ofstream f(file, mode);
    if (!f)
        return false;
    fill(f);
    f.close();
    if (!f)
        throw Domain::RelinkerException("Failed to write file: " + path);
    return true;
}

#ifndef _WIN32
bool moveOver(const std::filesystem::path& temporary, const std::filesystem::path& target, const std::filesystem::file_status& previous) {
    std::error_code error;
    if (std::filesystem::is_regular_file(previous))
        std::filesystem::permissions(temporary, previous.permissions() & std::filesystem::perms::all, error);
    if (!error)
        std::filesystem::rename(temporary, target, error);
    if (!error)
        return true;
    std::filesystem::remove(temporary, error);
    return false;
}
#endif

template<typename TFill>
void replaceFile(const std::string& path, const std::ios::openmode mode, const TFill& fill) {
    const auto target = NativePath(path);
#ifndef _WIN32
    std::error_code error;
    const auto previous = std::filesystem::symlink_status(target, error);
    auto temporary = target;
    temporary += ".tmp";
    if ((std::filesystem::is_regular_file(previous) || previous.type() == std::filesystem::file_type::not_found)
        && std::filesystem::symlink_status(temporary, error).type() == std::filesystem::file_type::not_found) {
        try {
            if (writeFile(temporary, path, mode, fill) && moveOver(temporary, target, previous))
                return;
        } catch (...) {
            std::filesystem::remove(temporary, error);
            throw;
        }
    }
#endif
    if (!writeFile(target, path, mode, fill))
        throw Domain::RelinkerException("Cannot open output file: " + path);
}

}

void FileWriter::Write(const std::string& path, const std::vector<std::uint8_t>& data) {
    replaceFile(path, std::ios::binary, [&data](std::ofstream& f) {
        f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    });
}

void FileWriter::Write(const std::string& path, const std::string& content) {
    replaceFile(path, std::ios::out, [&content](std::ofstream& f) { f << content; });
}

}
