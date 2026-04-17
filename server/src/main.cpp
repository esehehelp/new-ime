#include "socket_server.h"
#include "conversion_engine.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <csignal>
#include <unistd.h>

// TODO: Replace with protobuf serialization when build system is integrated.
// For now, use a simple text protocol for testing:
//   Request:  "CONVERT\tkana_input\tleft_context\tnum_candidates\n"
//   Response: "OK\tcandidate1\tscore1\tcandidate2\tscore2\t...\n"
//             or "ERROR\tmessage\n"

static newime::SocketServer* g_server = nullptr;

static void signal_handler(int /*sig*/) {
    if (g_server) g_server->shutdown();
}

static std::string get_socket_path() {
    std::string runtime_dir;
    if (const char* xdg = std::getenv("XDG_RUNTIME_DIR")) {
        runtime_dir = xdg;
    } else {
        runtime_dir = "/tmp";
    }
    return runtime_dir + "/new-ime-server." + std::to_string(getuid()) + ".sock";
}

int main(int /*argc*/, char** /*argv*/) {
    // Create mock inference engine
    auto backend = std::make_unique<newime::MockInferenceEngine>();
    newime::ConversionEngine engine(std::move(backend));

    // Set up signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // Create socket server with handler
    newime::SocketServer server([&engine](const std::string& request) -> std::string {
        // Simple text protocol for testing
        // Parse: "CONVERT\tkana\tcontext\tnum_candidates"
        auto tab1 = request.find('\t');
        if (tab1 == std::string::npos) return "ERROR\tmalformed request\n";

        std::string cmd = request.substr(0, tab1);

        if (cmd == "CONVERT") {
            auto tab2 = request.find('\t', tab1 + 1);
            auto tab3 = request.find('\t', tab2 + 1);

            std::string kana = request.substr(tab1 + 1, tab2 - tab1 - 1);
            std::string context = (tab2 != std::string::npos && tab3 != std::string::npos)
                ? request.substr(tab2 + 1, tab3 - tab2 - 1) : "";
            int num = 5;
            if (tab3 != std::string::npos) {
                try { num = std::stoi(request.substr(tab3 + 1)); } catch (...) {}
            }

            auto candidates = engine.convert(kana, context, num);

            std::string response = "OK";
            for (const auto& c : candidates) {
                response += "\t" + c.text + "\t" + std::to_string(c.score);
            }
            response += "\n";
            return response;
        }

        if (cmd == "SHUTDOWN") {
            if (g_server) g_server->shutdown();
            return "OK\n";
        }

        return "ERROR\tunknown command\n";
    });

    g_server = &server;

    std::string socket_path = get_socket_path();
    std::cerr << "new-ime-server: listening on " << socket_path << std::endl;

    if (!server.run(socket_path)) {
        std::cerr << "new-ime-server: failed to start" << std::endl;
        return 1;
    }

    return 0;
}
