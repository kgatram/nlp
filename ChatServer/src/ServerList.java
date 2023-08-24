import java.util.ArrayList;

/**
 * ServerList class maintains the list of all the clients connected to server.
 *
 * @author  220025456
 * @version "%I%, %G%"
 * @since JDK17
 *
 */
public class ServerList {
    private static ArrayList<ConnectionHandler> chandler = new ArrayList<ConnectionHandler>();
    private static ArrayList<String> channelList = new ArrayList<String>();
    private static ArrayList<String> channelNickList = new ArrayList<String>();

    /**
     * Get a list of all client handler object in the server.
     * @return returns ArrayList of ConnectionHandler objects.
     */
    public static synchronized ArrayList<ConnectionHandler> getChandler() {
        return chandler;
    }

    /**
     * Add a client handler object to server list.
     * @param input ConnectionHandler object.
     */
    public static synchronized void addChandler(ConnectionHandler input) {
            chandler.add(input);
    }

    /**
     * Remove a client handler object from server list.
     * @param input ConnectionHandler object.
     */
    public static synchronized void removeChandler(ConnectionHandler input) {
        chandler.remove(input);
    }

    /**
     * Get the list of all channel in the server.
     * @return returns ArrayList of string.
     */
    public static synchronized ArrayList<String> getChannelList() {
        return channelList;
    }

    /**
     * Add a channel to the server.
     * @param channel name of a channel.
     */
    public static synchronized void addChannelList(String channel) {
        if (!channelList.contains(channel)) {
            channelList.add(channel);
        }
    }

    /**
     * Remove a channel from the server only if no users are in the channel.
     * @param channel name of a channel.
     */
    public static synchronized void removeChannelList(String channel) {
        boolean found = false;
        for (String ch : channelNickList) {
            if (ch.startsWith(channel)) {
                found = true;
                break;
            }
        }
        if (!found) {
            channelList.remove(channel);
        }
    }

    /**
     * Get the list of channel and its user from the server.
     * @return returns ArrayList of string.
     */

    public static synchronized ArrayList<String> getChannelNickList() {
        return channelNickList;
    }

    /**
     * Get the list of channel and its user from the server.
     * @param channel name of channel.
     * @param nick nickname.
     */
    public static synchronized void addChannelNickList(String channel, String nick) {
        String channelNick = channel.trim() + "_" + nick.trim();
        if (!channelNickList.contains(channelNick)) {
            channelNickList.add(channelNick);
        }
    }

    /**
     * Remove a user and its channel from the server.
     * @param channel name of channel.
     * @param  nick  nickname.
     */
    public static synchronized void removeChannelNickList(String channel, String nick) {
        String channelNick = channel.trim() + "_" + nick.trim();
        channelNickList.remove(channelNick);

    }
}
