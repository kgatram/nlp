package stratego;

import stratego.pieces.Piece;

import java.util.Arrays;

/**
 * Game class with custom constructor, accessor methods.
 * Create a game for player 1 and player 2.
 * Creates a square for given row column
 * Checks for the winner.
 * @author 220025456
 */
public class Game {
    /**
     *  HEIGHT of board = 10.
     */
    public static final int HEIGHT = 10;
    /**
     *  WIDTH of board 10.
     */
    public static final int WIDTH = 10;
    /**
     *  Water rows in board.
     */
    public static final int[] WATER_ROWS = {4, 5};
    /**
     *  Water columns in board.
     */
    public static final int[] WATER_COLS = {2, 3, 6, 7};

    private Player p0, p1;

    /**
     * Constructor method for the class.
     *
     * @param    p0  Player object
     * @param    p1  Player object
     */
    public Game(Player p0, Player p1) {
        this.p0 = p0;
        this.p1 = p1;
        Square.initBoard();
    }


    /**
     * Get the player object by passing integer number 1 or 2.
     *
     * @param    playerNumber  player number either 1 or 2
     * @throws   IllegalArgumentException for input other than 1 or 2
     * @return   Player object
     */
    public Player getPlayer(int playerNumber) throws IllegalArgumentException {
        if (p0.getPlayerNumber() == playerNumber) {

            return p0;

        }
        else if (p1.getPlayerNumber() == playerNumber) {

            return p1;

        } else {

            throw new IllegalArgumentException();

        }

    }

    /**
     * Get winner of the game.
     *
     * @return   Player object
     */
    public Player getWinner() {
        if (p0.hasLost()) {

            return p1;

        } else if (p1.hasLost()) {

            return p0;

        } else {

            return null;
        }
    }

    /**
     * Get square object on the board if it exists else create a new square.
     *
     * @param    row  row number between 0 and 9.
     * @param    col  column number between 0 and 9.
     * @throws   IndexOutOfBoundsException when row or col not in range 0-9.
     * @return   Square object
     */
    public Square getSquare(int row, int col) throws IndexOutOfBoundsException {
        boolean isWater = false;

        if (row < 0 | row > 9 | col < 0 | col > 9) {

            throw new IndexOutOfBoundsException();
        }

        if (Arrays.binarySearch(WATER_ROWS, row) >= 0
                & Arrays.binarySearch(WATER_COLS, col) >= 0) {
            isWater = true;
        }

        if (Square.getSquare(row, col) == null) {

            return new Square(this, row, col, isWater);
        }
        else {

            return Square.getSquare(row, col);
        }

    }
}
