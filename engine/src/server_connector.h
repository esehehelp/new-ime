#pragma once

#include <string>
#include <vector>
#include <mutex>
#include <optional>

namespace newime {

/// Candidate returned from inference server.
struct Candidate {
    std::string text;     // Kanji-kana mixed result
    std::string reading;  // Hiragana reading
    float score = 0.0f;
};

/// IPC client for new-ime-server.
/// Communicates via Unix domain socket with length-prefixed protobuf messages.
class ServerConnector {
public:
    ServerConnector();
    ~ServerConnector();

    ServerConnector(const ServerConnector&) = delete;
    ServerConnector& operator=(const ServerConnector&) = delete;

    /// Connect to server. Starts server if not running.
    bool connect();

    /// Disconnect from server.
    void disconnect();

    /// Check if connected.
    bool is_connected() const { return fd_ >= 0; }

    /// Request kana-kanji conversion.
    std::vector<Candidate> convert(
        const std::string& kana_input,
        const std::string& left_context,
        int num_candidates = 5,
        bool use_refinement = false
    );

    /// Request incremental conversion (reuses encoder cache for prefix).
    std::vector<Candidate> convert_incremental(
        const std::string& kana_input,
        const std::string& left_context,
        const std::string& cached_prefix,
        int num_candidates = 5
    );

    /// Set surrounding text context.
    bool set_context(const std::string& surrounding_text, int cursor_position);

    /// Request server shutdown.
    bool shutdown_server();

private:
    /// Get socket path: ${XDG_RUNTIME_DIR}/new-ime-server.${UID}.sock
    static std::string socket_path();

    /// Start server process.
    bool start_server();

    /// Send request and receive response (protobuf bytes).
    std::optional<std::string> transact(const std::string& request_bytes);

    /// Send length-prefixed message.
    bool send_message(const std::string& data);

    /// Receive length-prefixed message.
    std::optional<std::string> recv_message();

    int fd_ = -1;
    std::mutex mutex_;
};

} // namespace newime
