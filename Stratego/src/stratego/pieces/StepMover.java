package stratego.pieces;

import stratego.Player;
import stratego.Square;

/**
 * Subclass of Piece and also Parent class of Miner and Spy with custom constructor.
 *
 * @author 220025456
 */
public class StepMover extends Piece {

    /**
     * Constructor method for the class.
     *
     * @param    owner  Player object
     * @param    square   Square object
     * @param    rank   integer value
     */
    public StepMover(Player owner, Square square, int rank) {
        super(owner, square, rank);
    }
}
