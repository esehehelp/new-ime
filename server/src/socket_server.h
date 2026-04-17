#pragma once

#include <string>
#include <functional>

namespace newime {

/// Unix domain socket server.
/// Accepts connections, reads length-prefixed messages, dispatches to handler.
class SocketServer {
public:
    /// Message handler: takes request bytes, returns response bytes.
    using Handler = std::function<std::string(const std::string&)>;

    explicit SocketServer(Handler handler);
    ~SocketServer();

    SocketServer(const SocketServer&) = delete;
    SocketServer& operator=(const SocketServer&) = delete;

    /// Start listening. Blocks until shutdown() is called.
    /// @param socket_path Path for Unix domain socket.
    bool run(const std::string& socket_path);

    /// Signal the server to stop.
    void shutdown();

private:
    bool setup_socket(const std::string& path);
    void accept_loop();
    void handle_client(int client_fd);

    Handler handler_;
    int listen_fd_ = -1;
    bool running_ = false;
};

} // namespace newime
