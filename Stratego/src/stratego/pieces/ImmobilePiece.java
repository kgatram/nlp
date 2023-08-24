package stratego.pieces;

import stratego.Player;
import stratego.Square;

/**
 * Subclass of Piece with custom constructor.
 *
 * @author 220025456
 */
public abstract class ImmobilePiece extends Piece {

    /**
     * Constructor method for the class.
     *
     * @param    owner  Player object
     * @param    square   Square object
     * @param    rank   integer value
     */
    public ImmobilePiece(Player owner, Square square, int rank) {
        super(owner, square, rank);
    }

}
