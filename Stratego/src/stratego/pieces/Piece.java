package stratego.pieces;

import stratego.CombatResult;
import stratego.Game;
import stratego.Player;
import stratego.Square;

import java.util.ArrayList;
import java.util.List;

/**
 * Parent class with custom constructor, accessor methods for all the pieces of stratego.
 *
 * @author 220025456
 */
public class Piece {
    private Player owner;
    private Square square;
    private int rank;
    private boolean isActive;

    /**
     * Constructor method for the class.
     *
     * @param    owner  Player object
     * @param    square   Square object
     * @param    rank   integer value
     */
    public Piece(Player owner, Square square, int rank) {
        this.owner = owner;
        this.square = square;
        this.rank = rank;
        this.isActive = true;
        square.placePiece(this, true);
    }

    /**
     * Return square occupied by piece. If piece is killed return null.
     *
     * @return   Square object
     */
    public Square getSquare() {
        if (!isActive) {

            return null;
        }

        return square;
    }

    /**
     * Return player who owns the piece.
     *
     * @return   Player object
     */
    public Player getOwner() {
        return owner;
    }

    /**
     * Return rank of the piece.
     *
     * @return   int between 2 and 10 and 0 for non-ranked pieces.
     */
    public int getRank() {
        if (rank < 2) {

            return 0;
        }

        return rank;
    }

    /**
     * Set rank of the piece.
     *
     * @param  rank integer value.
     */
    public void setRank(int rank) {
        this.rank = rank;
    }

    /**
     * Set the piece active or inactive.
     *
     * @param    active true , inactive false
     */
    public void setActive(boolean active) {
        isActive = active;
    }

    /**
     * Shows the result WIN, DRAW, LOSE when a piece attacks another piece.
     *
     * @param    targetPiece Piece Object
     * @return   CombatResult WIN, DRAW, LOSE
     */
    public CombatResult resultWhenAttacking(Piece targetPiece) {
// if not miner then attack on bomb draws.
        if ((targetPiece instanceof Bomb)
                && !(this instanceof Miner)) {
            return CombatResult.DRAW;
        }

        if (this.getRank() > targetPiece.getRank()) {
            return CombatResult.WIN;
        }
        else if (this.getRank() < targetPiece.getRank()) {
            return CombatResult.LOSE;
        }
        return CombatResult.DRAW;
    }

    /**
     * As per rule of the game, decides who survives or get killed when a piece attacks a target piece.
     *
     * @param    targetSquare Square Object
     */

    public void attack(Square targetSquare) {

// if not miner then set the rank of bomb same as piece.
        if ((targetSquare.getPiece() instanceof Bomb)
                && !(this instanceof Miner)) {
            targetSquare.getPiece().setRank(this.rank);
        }

        if (targetSquare.getPiece() instanceof Flag) {
            targetSquare.getPiece().getOwner().loseGame();  // set the winner
        }

        if (this.getRank() > targetSquare.getPiece().getRank()) {

//If attacks on lower rank, kill lower rank
            targetSquare.getPiece().setActive(false);
// higher takes the new space.
            this.move(targetSquare);

        } else if (this.getRank() < targetSquare.getPiece().getRank()) {

            this.getSquare().removePiece();  //remove lower rank
            this.setActive(false);           //kill piece

        } else {
//If the two pieces have the same rank then they are both destroyed.
            this.getSquare().removePiece();
            this.setActive(false);           //kill piece
            targetSquare.getPiece().setActive(false);
            targetSquare.removePiece();
        }

    }

    /**
     * Move the piece to target squares.
     *
     * @param toSquare Square object
     */
    public void move(Square toSquare) {
        square.removePiece();
        square = toSquare;
        square.placePiece(this, true);
    }

    /**
     * Get all possible squares which are legal move for a piece.
     *
     * @return   List of Square objects.
     */
    public List<Square> getLegalMoves() {
        ArrayList<Square> legalMoves = new ArrayList<>();

        int column = square.getCol();
        int rows = square.getRow();
        Game gam = square.getGame();

        for (int i = -1; i <= 1; i += 2) {
            try {

                Square newSquare = gam.getSquare(rows, column + i);
                if (newSquare.canBeEntered()) {

                    legalMoves.add(newSquare);
                }

                newSquare = gam.getSquare(rows + i, column);
                if (newSquare.canBeEntered()) {

                    legalMoves.add(newSquare);
                }

            } catch (IndexOutOfBoundsException e) {
                continue;
            }
        }

        return legalMoves;
    }

    /**
     * Get all possible squares which are legal attack for a piece.
     *
     * @return   List of Square objects.
     */
    public List<Square> getLegalAttacks() {
        ArrayList<Square> legalAttacks = new ArrayList<>();

        int column = square.getCol();
        int rows = square.getRow();
        Game gam = square.getGame();

        // look adjacent pieces for attack
        for (int i = -1; i <= 1; i += 2) {
            try {

                Square newSquare = gam.getSquare(rows, column + i);
                if (newSquare.getPiece() != null
                        && this.getOwner() != newSquare.getPiece().getOwner()) {

                    legalAttacks.add(newSquare);
                }

                newSquare = gam.getSquare(rows + i, column);
                if (newSquare.getPiece() != null
                        && this.getOwner() != newSquare.getPiece().getOwner()) {

                    legalAttacks.add(newSquare);
                }

            } catch (IndexOutOfBoundsException e) {
                continue;
            }
        }

        return legalAttacks;
    }

    /**
     * Removes a piece from a square.
     *
     */
    public void beCaptured() {
        square.removePiece();
        this.setActive(false);
    }

}
