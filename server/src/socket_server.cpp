#include "socket_server.h"

#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <csignal>

namespace newime {

SocketServer::SocketServer(Handler handler)
    : handler_(std::move(handler)) {}

SocketServer::~SocketServer() {
    shutdown();
}

bool SocketServer::setup_socket(const std::string& path) {
    // Remove existing socket file
    unlink(path.c_str());

    listen_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (listen_fd_ < 0) return false;

    struct sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);

    if (bind(listen_fd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        close(listen_fd_);
        listen_fd_ = -1;
        return false;
    }

    // Restrict permissions to owner only
    chmod(path.c_str(), 0600);

    if (listen(listen_fd_, 5) < 0) {
        close(listen_fd_);
        listen_fd_ = -1;
        return false;
    }

    return true;
}

bool SocketServer::run(const std::string& socket_path) {
    if (!setup_socket(socket_path)) return false;

    running_ = true;
    accept_loop();

    // Cleanup
    close(listen_fd_);
    listen_fd_ = -1;
    unlink(socket_path.c_str());
    return true;
}

void SocketServer::accept_loop() {
    while (running_) {
        // Use select() with timeout so we can check running_ flag
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(listen_fd_, &rfds);

        struct timeval tv{};
        tv.tv_sec = 1;
        tv.tv_usec = 0;

        int ret = select(listen_fd_ + 1, &rfds, nullptr, nullptr, &tv);
        if (ret <= 0) continue;

        int client_fd = accept(listen_fd_, nullptr, nullptr);
        if (client_fd < 0) continue;

        handle_client(client_fd);
        close(client_fd);
    }
}

void SocketServer::handle_client(int client_fd) {
    while (running_) {
        // Read length prefix (4 bytes, big-endian)
        uint8_t header[4];
        ssize_t n = read(client_fd, header, 4);
        if (n != 4) return;

        uint32_t len = (static_cast<uint32_t>(header[0]) << 24) |
                       (static_cast<uint32_t>(header[1]) << 16) |
                       (static_cast<uint32_t>(header[2]) << 8) |
                       static_cast<uint32_t>(header[3]);

        if (len > 1024 * 1024) return;  // 1MB sanity limit

        // Read message body
        std::string request(len, '\0');
        size_t received = 0;
        while (received < len) {
            n = read(client_fd, request.data() + received, len - received);
            if (n <= 0) return;
            received += n;
        }

        // Dispatch to handler
        std::string response = handler_(request);

        // Send length-prefixed response
        uint32_t resp_len = static_cast<uint32_t>(response.size());
        uint8_t resp_header[4] = {
            static_cast<uint8_t>((resp_len >> 24) & 0xFF),
            static_cast<uint8_t>((resp_len >> 16) & 0xFF),
            static_cast<uint8_t>((resp_len >> 8) & 0xFF),
            static_cast<uint8_t>(resp_len & 0xFF),
        };

        if (write(client_fd, resp_header, 4) != 4) return;
        if (write(client_fd, response.data(), response.size()) !=
            static_cast<ssize_t>(response.size())) return;
    }
}

void SocketServer::shutdown() {
    running_ = false;
    if (listen_fd_ >= 0) {
        ::shutdown(listen_fd_, SHUT_RDWR);
    }
}

} // namespace newime
