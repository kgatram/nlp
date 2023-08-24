package stratego;

import stratego.pieces.Piece;

import java.util.Arrays;

import static stratego.Game.HEIGHT;
import static stratego.Game.WIDTH;

/**
 * Square class with custom constructor, accessor methods.
 * Create a square object for a given position row and column.
 *
 * @author 220025456
 */

public class Square {
    /**
     * board will track the location of piece in a game.
     */
    private static Piece[][] board = new Piece[HEIGHT][WIDTH];

    /**
     * squares will recycle square objects throughout the game.
     */
    private static Square[][] squares = new Square[HEIGHT][WIDTH];
    private Game game;
    private int row, col;
    private boolean isWater;

    /**
     * Constructor method for the class.
     *
     * @param    game  game object
     * @param    row   integer row location
     * @param    col   integer column location
     * @param    isWater  location has water
     */
    public Square(Game game, int row, int col, boolean isWater) {
        this.game = game;
        this.row = row;
        this.col = col;
        this.isWater = isWater;
        squares[row][col] = this;
    }

    /**
     * Place a piece on a given square if the square is not occupied.
     *
     * @param    piece  Piece object to place
     * @throws   IllegalArgumentException if the square is already occupied
     */
    public void placePiece(Piece piece) throws IllegalArgumentException {
        if (board[row][col] == null) {

            board[row][col] = piece;
        }
        else {

            throw new IllegalArgumentException("square already occupied");
        }
    }

    /**
     * Place a piece on a given square.
     *
     * @param    piece  Piece object to place on board
     * @param    ignoreCheck  perform no checks
     */
    public void placePiece(Piece piece, boolean ignoreCheck) {
            board[row][col] = piece;
    }

    /**
     * Get a piece occupying the square, if empty return null.
     *
     * @return    Piece object occupying the square else return null.
     */
    public Piece getPiece() {
        if (row >= 0 && col >= 0) {

            return board[row][col];
        }

        return null;
    }

    /**
     * Remove piece from the square, if empty do nothing.
     *
     */
    public void removePiece() {
        if (row >= 0 && col >= 0) {
            board[row][col] = null;
        }
    }

    /**
     * For new game initialise the board.
     *
     */
    public static void initBoard() {

        for (Piece[] pieces : board) {
            Arrays.fill(pieces, null);
        }
    }

    /**
     * get a square for the game.
     *
     * @param row int between 0 and 9.
     * @param col int between 0 and 9.
     * @return  Square object.
     */
    public static Square getSquare(int row, int col) {
        return squares[row][col];
    }

    /**
     * return Game object.
     *
     * @return    Game object
     */
    public Game getGame() {
        return game;
    }

    /**
     * Accessor method to return row value of a square object.
     *
     * @return    row integer value between 0 and 9.
     */
    public int getRow() {
        return row;
    }

    /**
     * Accessor method to return column value of a square object.
     *
     * @return    col integer value between 0 and 9.
     */
    public int getCol() {
        return col;
    }

    /**
     * Setter method to set row value of a square object.
     *
     * @param row int value for row
     */

    public void setRow(int row) {
        this.row = row;
    }

    /**
     * Setter method to set column value of a square object.
     *
     * @param col int value for column
     */

    public void setCol(int col) {
        this.col = col;
    }

    /**
     * Check if a square is available for move.
     *
     * @return  true is the square is not occupied.
     */
    public boolean canBeEntered() {
        if (isWater) {

            return false;
        }
        else {
            return board[row][col] == null;
        }
    }
}
