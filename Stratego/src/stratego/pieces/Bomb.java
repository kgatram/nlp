package stratego.pieces;

import stratego.Player;
import stratego.Square;

import java.util.ArrayList;
import java.util.List;

/**
 * Subclass of ImmobilePiece with custom constructor.
 * Overrides getLegalMoves and getLegalAttacks method.
 *
 * @author 220025456
 */
public class Bomb extends ImmobilePiece {

    /**
     * Constructor method for the class.
     *
     * @param    owner  Player object
     * @param    square   Square object
     */
    public Bomb(Player owner, Square square) {
        super(owner, square, 0);
    }

    /**
     * Bomb has no moves, return empty List of square.
     *
     * @return   List of Square objects.
     */

    public List<Square> getLegalMoves() {
        ArrayList<Square> legalMoves = new ArrayList<Square>();
        return legalMoves;
    }

    /**
     * Bomb can't attack, return empty List of square.
     *
     * @return   List of Square objects.
     */

    public List<Square> getLegalAttacks() {
        ArrayList<Square> legalAttacks = new ArrayList<Square>();
        return legalAttacks;
    }
}
