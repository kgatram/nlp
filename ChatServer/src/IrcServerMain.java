/**
 * This is an executable class to initiate chat server.
 * Chat server follows simplified version of the IRC protocol. It accept connections from clients using Java’s
 * Socket and ServerSocket objects, keeping multiple connections alive at once and sending the appropriate updates
 * to each one in real time.
 * Usage from CLI: java IrcServerMain &lt;server_name&gt; &lt;port&gt;
 *
 * @author  220025456
 * @version "%I%, %G%"
 * @since JDK17
 *
 */

public class IrcServerMain {
    /**
     * Main method for chat server. Get following arguments from command line and passes argument to
     * ServerCxn class for Socket connection.
     *
     * @param args The first argument is the name of the server.
     *             The second argument is port number having digits.
     *
     */

    public static void main(String[] args) {
        try {
            ServerCxn server = new ServerCxn(args[0], Integer.parseInt(args[1]));

        } catch (Exception exp) {
            System.out.println("Usage: java IrcServerMain <server_name> <port>");
        }
    }
}
