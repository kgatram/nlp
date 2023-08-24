package stratego.pieces;

import stratego.Player;
import stratego.Square;

/**
 * Subclass of StepMover with custom constructor.
 *
 * @author 220025456
 */
public class Miner extends StepMover {

    /**
     * Constructor method for the class.
     *
     * @param    owner  Player object
     * @param    square   Square object
     */
    public Miner(Player owner, Square square) {
        super(owner, square, 3);
    }

}
