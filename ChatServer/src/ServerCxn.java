import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;

/**
 * ServerCxn class gets arguments from caller, and it accepts connections from clients using Java’s
 * Socket objects. It creates a connection handler object to keeping multiple connections alive using threading and
 * sends appropriate updates to each connected clients in real time
 *
 * @author  220025456
 * @version "%I%, %G%"
 * @since JDK17
 *
 */
public class ServerCxn {
    private ServerSocket server_socket;
    /**
     * Listens to the port and wait for a connection from client. Once a client connects it initiates a handler process
     * using multi-threading.
     *
     * @param servName name of the server.
     * @param port     port number having digits.
     *
     */

    public ServerCxn(String servName, int port) {
        try {
            server_socket = new ServerSocket(port);
            while (true) {
                Socket clientCxn = server_socket.accept();
                ConnectionHandler handler = new ConnectionHandler(servName, clientCxn);
                handler.start();
            }

        } catch (IOException e) {
            System.out.println(e.getMessage());
        }

    }

}
