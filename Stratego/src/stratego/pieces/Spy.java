package stratego.pieces;

import stratego.CombatResult;
import stratego.Player;
import stratego.Square;

/**
 * Subclass of StepMover with custom constructor.
 * Overrides resultWhenAttacking method.
 *
 * @author 220025456
 */
public class Spy extends StepMover {
    private static final int MARSHAL = 10;

    /**
     * Constructor method for the class.
     *
     * @param    owner  Player object
     * @param    square   Square object
     */
    public Spy(Player owner, Square square) {
        super(owner, square, 1);
    }

    /**
     * Shows the result WIN, DRAW, LOSE when a piece attacks another piece.
     * If any piece attacks a Spy, the Spy is destroyed. However, if a Spy attacks a Marshal,
     * then the Marshal is destroyed instead. If a Spy attacks any piece other than a Marshal or a
     * Flag, then the Spy is destroyed
     *
     * @param    targetPiece Piece Object
     * @return   CombatResult WIN, DRAW, LOSE
     */

    public CombatResult resultWhenAttacking(Piece targetPiece) {

// bomb draws with spy.
        if (targetPiece instanceof Bomb) {
            return CombatResult.DRAW;
        }

// Spy attacks a Marshal, then the Marshal is destroyed.
        if (this.getRank() > targetPiece.getRank()
                || targetPiece.getRank() == MARSHAL) {
            return CombatResult.WIN;
        }
// If any piece attacks a Spy, the Spy is destroyed.
        else if (this.getRank() < targetPiece.getRank()) {
            return CombatResult.LOSE;
        }

        return CombatResult.DRAW;
    }
}
