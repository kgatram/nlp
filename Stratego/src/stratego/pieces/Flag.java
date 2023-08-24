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
public class Flag extends ImmobilePiece {

    /**
     * Constructor method for the class.
     *
     * @param    owner  Player object
     * @param    square   Square object
     */
    public Flag(Player owner, Square square) {
        super(owner, square, -1);
    }

    /**
     * Flag has no moves, return empty List of square.
     *
     * @return   List of Square objects.
     */
    public List<Square> getLegalMoves() {
        ArrayList<Square> legalMoves = new ArrayList<Square>();
        return legalMoves;
    }

    /**
     * Flag can't attack, return empty List of square.
     *
     * @return   List of Square objects.
     */
    public List<Square> getLegalAttacks() {
        ArrayList<Square> legalAttacks = new ArrayList<Square>();
        return legalAttacks;
    }

}
