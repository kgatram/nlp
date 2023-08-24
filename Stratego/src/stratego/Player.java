package stratego;

/**
 * Player class with custom constructor, accessor methods.
 * Create a player with a name and player number {0, 1}.
 *
 * @author 220025456
 */
public class Player {
    private String name;
    private int playerNumber;
    private boolean lost;

    /**
     * Constructor method for the class.
     *
     * @param    name  string
     * @param    playerNumber   integer value
     */
    public Player(String name, int playerNumber) {
        this.name = name;
        this.playerNumber = playerNumber;
        this.lost = false;
    }

    /**
     * Returns the name of the player.
     *
     * @return   String containing name of player
     */
    public String getName() {
        return name;
    }

    /**
     * Returns the number of the player.
     *
     * @return   integer containing number of player
     */
    public int getPlayerNumber() {
        return playerNumber;
    }

    /**
     * Set the player to lose the game.
     *
     */
    public void loseGame() {
        lost = true;
    }

    /**
     * Return true if player has lost, false otherwise.
     *
     * @return boolean true if player has lost
     */
    public boolean hasLost() {
        return lost;
    }

}
