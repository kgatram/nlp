import java.io.BufferedReader;
import java.io.InputStream;
import java.io.PrintWriter;
import java.io.InputStreamReader;
import java.io.IOException;
import java.net.Socket;
import java.time.LocalDateTime;
import java.util.regex.Pattern;

/**
 * ConnectionHandler class runs in multithreaded environment.  Accept commands from the client and take appropriate action.
 *
 * @author  220025456
 * @version "%I%, %G%"
 * @since JDK17
 *
 */

public class ConnectionHandler extends Thread {
    private BufferedReader readr;
    private final Socket clientCxn;
    private final String servName;
    private String nickname = null;
    private String username = null;
    private String real_name = " ";
    private String channel = null;
    private InputStream incoming;
    private PrintWriter outgoing;

    private final Pattern space = Pattern.compile(" ");

    /**
     * Constructor for ConnectionHandler class.
     * Attach input and output stream to ConnectionHandler object.
     *
     * @param  servName Server name
     * @param  clientCxn client socket object
     *
     */

    public ConnectionHandler(String servName, Socket clientCxn) {
        this.servName = servName;
        this.clientCxn = clientCxn;
        try {
            incoming = clientCxn.getInputStream();
            readr = new BufferedReader(new InputStreamReader(incoming));
            outgoing = new PrintWriter(clientCxn.getOutputStream(), true);
        } catch (IOException exp) {
            System.out.println("ConnectionHandler error");
        }
    }

    /**
     * initiate a thread for ConnectionHandler object.
     *
     */
    public void run() {
        try {
            handleRequest();
        } catch (Exception exp) {
            System.out.println("Closing all connections...");
            closeCxn();
        }
    }

    /**
     * Process all the requests from Client. Reads the command from input stream and reply back to output stream.
     * @throws IOException raise exception to terminate connection.
     */
    private void handleRequest() throws IOException {
        String data;
        String[] command;

        while (true) {
            data = readr.readLine();
            command = space.split(data, 2);

            switch (command[0]) {
                case "PRIVMSG" -> sendMessage(data);
                case "NICK" -> setNickname(command);
                case "USER" -> setUser(data);
                case "JOIN" -> joinChannel(command);
                case "PART" -> partChannel(command);
                case "NAMES" -> getNames(command);
                case "LIST" -> getList();
                case "INFO" -> getInfo();
                case "TIME" -> getTime();
                case "PING" -> replyPing(command);
                case "QUIT" -> quit();
                default -> {
                }
            }
        }
    }

    /**
     * Check for client registration.
     * If USER message is valid and nickname has already been set, then the server should store all these details and
     * regard the client as registered.
     */
    private boolean notRegistered() {
        if (nickname == null
                || username == null
                || real_name.equals(" ")) {
            outgoing.println(":" + servName + " 400 * :You need to register first");
            return true;
        }
        return false;
    }

    /**
     * Process PRIVMSG command.
     * @param input take command string as input.
     */
    private void sendMessage(String input) {
        final int length_THREE = 3;
        String[] data = space.split(input, length_THREE);
        boolean sentToChannel = false, sentToUser = false;
        String nick = nickname == null ? "*" : nickname;

        if (notRegistered()) {
            return;
        }

        if (data.length < length_THREE) {
            outgoing.println(":" + servName + " 400 " + nick + " :Invalid arguments to PRIVMSG command");
            return;
        }

        if (ServerList.getChannelList().contains(data[1])) {
            //        broadcast message to the channel.
            for (ConnectionHandler ch : ServerList.getChandler()) {
                if (ch.channel != null
                        && ch.channel.equals(data[1])) {
                    ch.outgoing.println(":" + this.nickname + " " + data[0] + " " + data[1] + " " + data[2]);
                    sentToChannel = true;
                }
            }
        } else {
            for (ConnectionHandler ch : ServerList.getChandler()) {
                if (ch.nickname.equals(data[1])) {
                    ch.outgoing.println(":" + this.nickname + " " + data[0] + " " + data[1] + " " + data[2]);
                    sentToUser = true;
                    break;
                }
            }
        }

        if (!sentToChannel && !sentToUser) {
            outgoing.println(":" + servName + " 400 * :No user exists with that name");
        }

    }

    /**
     * Process NICK command.
     * @param data take command string as input.
     */
    private void setNickname(String[] data) {
        String nick;

        nick = data.length < 2 ? " " : data[1];

//  A valid nickname has 1–9 characters, and contains
//  only letters, numbers and underscores. It cannot start with a number.

        if (!Pattern.matches("^[a-zA-Z_]\\w{0,8}$", nick)) {
            String reply = ":" + servName + " 400 * :Invalid nickname";
            outgoing.println(reply);
            return;
        }

        nickname = data[1];
    }

    /**
     * Process USER command.
     * @param input take command string as input.
     */
    private void setUser(String input) {
        String reply, user, name;
        final int element_FOUR = 4;
        final int length_FIVE = 5;
        String[] data = space.split(input, length_FIVE);
        String nick = nickname == null ? "*" : nickname;


        if (nickname == null) {
            outgoing.println(":" + servName + " 400 * :use NICK before USER");
            return;
        }

        if (data.length < length_FIVE) {
            outgoing.println(":" + servName + " 400 " + nick + " :Not enough arguments");
            return;
        }

        user = data[1];
        name = data[element_FOUR].replaceFirst(":", " ").trim();

//        username should not include spaces.
        if (!Pattern.matches("\\S+", user)
                | !data[element_FOUR].startsWith(":")
                | name.equals("")) {
            reply = ":" + servName + " 400 " + nick + " :Invalid arguments to USER command";
            outgoing.println(reply);
            return;
        }

        if (username != null && !real_name.equals(" ")) {
            outgoing.println(":" + servName + " 400 " + nick  + " :You are already registered");
            return;
        }

        username = user;
        real_name = name;
        reply = ":" + servName + " 001 " + nickname +  " :Welcome to the IRC network, " + nickname;
        outgoing.println(reply);

//        A registered client can send and receive private messages, and join and leave channels.
//        Add entry with null channel.
        ServerList.addChandler(this);

    }

    /**
     * Check for valid channel.
     * @param in_channel takes a channel name as input.
     */
    private boolean invalidChannel(String in_channel) {
        String nick = nickname == null ? "*" : nickname;

//        A channel must be # followed by any number of letters, numbers and underscores.
        if (!Pattern.matches("#[a-zA-Z0-9_]+", in_channel)) {
            String reply = ":" + servName + " 400 " + nick + " :Invalid channel name";
            outgoing.println(reply);
            return true;
        }
        return false;
    }

    /**
     * Process JOIN command.
     * @param data take command string as input.
     */
    private void joinChannel(String[] data) {

        String in_channel;

        if (notRegistered()) {
            return;
        }

        in_channel = (data.length < 2) ? " " : data[1];

        if (invalidChannel(in_channel)) {
            return;
        }

//      Maintain server list. Remove entry for previous channel if not null.
        ServerList.removeChandler(this);
        if (channel != null) {
            ServerList.removeChannelNickList(channel, nickname);
        }

//      Add new entry for new channel.
        channel = in_channel;
        ServerList.addChannelNickList(channel, nickname);
        ServerList.addChannelList(channel);
        ServerList.addChandler(this);

//        broadcast message to the channel.
        broadcastChannel(data[0]);
    }

    /**
     * Broadcast message to all the users in a channel.
     * @param command take command string as input.
     */
    private void broadcastChannel(String command) {

        for (ConnectionHandler ch : ServerList.getChandler()) {
            if (ch.channel != null
                    && ch.channel.equals(this.channel)) {
                ch.outgoing.println(":" + this.nickname + " " + command + " " + this.channel);
            }
        }
    }

    /**
     * Check if a channel exists?
     * @param in_channel take channel input.
     */
    private boolean notAChannel(String in_channel) {
        String nick = nickname == null ? "*" : nickname;

        if (!ServerList.getChannelList().contains(in_channel)) {
            outgoing.println(":" + servName + " 400 " + nick + " :No channel exists with that name");
            return true;
        }
        return false;
    }

    /**
     * Process PART command. when that user wishes to leave a channel they are in.
     * @param data take command string as input.
     */
    private void partChannel(String[] data) {

        String in_channel;

        if (notRegistered()) {
            return;
        }

        in_channel = (data.length < 2) ? " " : data[1];

//        channel does not exist
        if (notAChannel(in_channel)) {
            return;
        }

//        if user not in the channel no action taken.
        if (!in_channel.equals(channel)) {
            return;
        }

//          broadcast message to the channel.
        broadcastChannel(data[0]);

//        Maintain server list. Remove entries for parting channel
        ServerList.removeChandler(this);
        ServerList.removeChannelNickList(in_channel, nickname);
        ServerList.removeChannelList(in_channel);

        channel = null;
        ServerList.addChandler(this);

    }

    /**
     * Process NAMES command. Get the nicknames of all users in a given channel.
     * @param data take command string as input.
     */
    private void getNames(String[] data) {
        String reply, in_channel;
        Pattern underscore = Pattern.compile("_");

        if (notRegistered()) {
            return;
        }

        in_channel = (data.length < 2) ? " " : data[1];

//        channel does not exist
        if (notAChannel(in_channel)) {
            return;
        }

        reply = ":" + servName + " 353 " + nickname + " = " + in_channel + " :";
        StringBuffer sb = new StringBuffer(reply);

        for (String ch : ServerList.getChannelNickList()) {
            String[] str = underscore.split(ch);
            if (str[0].equals(in_channel)) {
                sb.append(str[1]);
                sb.append(" ");
            }
        }
        outgoing.println(sb);
    }

    /**
     * Process LIST command. Get the names of all channels on the server.
     */
    private void getList() {
        String reply;

        if (notRegistered()) {
            return;
        }

        for (String ch : ServerList.getChannelList()) {
            reply = ":" + servName + " 322 " + nickname + " " + ch;
            outgoing.println(reply);
        }

        reply = ":" + servName + " 323 " + nickname + " :End of LIST";
        outgoing.println(reply);
    }

    /**
     * Process INFO command. Get basic information about the server.
     */
    private void getInfo() {
        String reply;

        if (nickname == null) {
            reply = ":" + servName + " 371 * :" + "U r connected to Chat Server 220025456";
        } else {
            reply = ":" + servName + " 371 * :" + nickname + " is connected to Chat Server 220025456 channel " + channel;
        }

        outgoing.println(reply);
    }

    /**
     * Process TIME command. Get date-time from server.
     */
    private void getTime() {
        String reply = ":" + servName + " 391 * :" + LocalDateTime.now().toString();
        outgoing.println(reply);
    }
    private void replyPing(String[] data) {

        if (data.length < 2) {
            outgoing.println("PONG");
            return;
        }

        outgoing.println("PONG " + data[1]);
    }

    /**
     * Process QUIT command. The user wants to end their connection to the server.
     * @throws IOException to terminate connection from server.
     */
    private void quit() throws IOException {
//        message all connected client.
        for (ConnectionHandler ch : ServerList.getChandler()) {
                ch.outgoing.println(":" + nickname + " QUIT ");
            }
//        remove from server list.
        ServerList.removeChandler(this);

        if (channel != null) {
            ServerList.removeChannelNickList(channel, nickname);
            ServerList.removeChannelList(channel);
        }

        throw new IOException();
    }

    /**
     * Close input stream, output stream and terminate the thread.
     */
    private void closeCxn() {
        try {
            readr.close();
            incoming.close();
            clientCxn.close();
        } catch (Exception exp) {
            System.out.println("Error closing connection.");
        }

    }
}
