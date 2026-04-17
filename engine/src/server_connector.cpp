#include "server_connector.h"

#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>

namespace newime {

static constexpr int MAX_CONNECT_RETRIES = 8;
static constexpr int RETRY_INTERVAL_MS = 250;
static constexpr int RECV_TIMEOUT_SEC = 5;

ServerConnector::ServerConnector() = default;

ServerConnector::~ServerConnector() {
    disconnect();
}

std::string ServerConnector::socket_path() {
    std::string runtime_dir;
    if (const char* xdg = std::getenv("XDG_RUNTIME_DIR")) {
        runtime_dir = xdg;
    } else {
        runtime_dir = "/tmp";
    }
    return runtime_dir + "/new-ime-server." + std::to_string(getuid()) + ".sock";
}

bool ServerConnector::start_server() {
    // Fork and exec new-ime-server
    pid_t pid = fork();
    if (pid == 0) {
        // Child: exec server
        setsid();
        execlp("new-ime-server", "new-ime-server", nullptr);
        _exit(1);
    }
    return pid > 0;
}

bool ServerConnector::connect() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (fd_ >= 0) return true;

    for (int attempt = 0; attempt < MAX_CONNECT_RETRIES; attempt++) {
        fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd_ < 0) return false;

        struct sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        std::string path = socket_path();
        strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);

        if (::connect(fd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) == 0) {
            // Set receive timeout
            struct timeval tv{};
            tv.tv_sec = RECV_TIMEOUT_SEC;
            setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
            return true;
        }

        close(fd_);
        fd_ = -1;

        // Start server on first failure
        if (attempt == 0) {
            start_server();
        }

        usleep(RETRY_INTERVAL_MS * 1000);
    }

    return false;
}

void ServerConnector::disconnect() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
}

bool ServerConnector::send_message(const std::string& data) {
    // Length prefix: 4 bytes, big-endian
    uint32_t len = static_cast<uint32_t>(data.size());
    uint8_t header[4] = {
        static_cast<uint8_t>((len >> 24) & 0xFF),
        static_cast<uint8_t>((len >> 16) & 0xFF),
        static_cast<uint8_t>((len >> 8) & 0xFF),
        static_cast<uint8_t>(len & 0xFF),
    };

    if (write(fd_, header, 4) != 4) return false;
    if (write(fd_, data.data(), data.size()) != static_cast<ssize_t>(data.size())) return false;
    return true;
}

std::optional<std::string> ServerConnector::recv_message() {
    uint8_t header[4];
    ssize_t n = read(fd_, header, 4);
    if (n != 4) return std::nullopt;

    uint32_t len = (static_cast<uint32_t>(header[0]) << 24) |
                   (static_cast<uint32_t>(header[1]) << 16) |
                   (static_cast<uint32_t>(header[2]) << 8) |
                   static_cast<uint32_t>(header[3]);

    if (len > 1024 * 1024) return std::nullopt;  // 1MB sanity limit

    std::string data(len, '\0');
    size_t received = 0;
    while (received < len) {
        n = read(fd_, data.data() + received, len - received);
        if (n <= 0) return std::nullopt;
        received += n;
    }

    return data;
}

std::optional<std::string> ServerConnector::transact(const std::string& request_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (fd_ < 0 && !connect()) return std::nullopt;
    if (!send_message(request_bytes)) {
        disconnect();
        return std::nullopt;
    }
    auto response = recv_message();
    if (!response) {
        disconnect();
    }
    return response;
}

// Stub implementations - actual protobuf serialization will be added
// when protobuf code generation is integrated into the build.

std::vector<Candidate> ServerConnector::convert(
    const std::string& /*kana_input*/,
    const std::string& /*left_context*/,
    int /*num_candidates*/,
    bool /*use_refinement*/)
{
    // TODO: serialize ConvertRequest, transact, deserialize CandidatesResult
    return {};
}

std::vector<Candidate> ServerConnector::convert_incremental(
    const std::string& /*kana_input*/,
    const std::string& /*left_context*/,
    const std::string& /*cached_prefix*/,
    int /*num_candidates*/)
{
    // TODO: serialize ConvertIncrementalRequest, transact, deserialize
    return {};
}

bool ServerConnector::set_context(
    const std::string& /*surrounding_text*/,
    int /*cursor_position*/)
{
    // TODO: serialize SetContextRequest, transact
    return false;
}

bool ServerConnector::shutdown_server() {
    // TODO: serialize ShutdownRequest, transact
    return false;
}

} // namespace newime
