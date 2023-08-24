package stratego.pieces;

import stratego.Game;
import stratego.Player;
import stratego.Square;

import java.util.ArrayList;
import java.util.List;

/**
 * Subclass of Piece with custom constructor.
 * Overrides getLegalMoves()
 *
 * @author 220025456
 */
public class Scout extends Piece {

    /**
     * Constructor method for the class.
     *
     * @param    owner  Player object
     * @param    square   Square object
     */
    public Scout(Player owner, Square square) {
        super(owner, square, 2);
    }

    /**
     * Get all possible squares which are legal move for a Scout
     * The six Scout pieces can move more than one square: they can move as many open spaces as
     * they like in a single direction (forward, backward or sideways), but cannot jump over pieces. A
     * Scout attacks like a normal piece.
     *
     * @return   List of Square objects.
     */
    public List<Square> getLegalMoves() {
        ArrayList<Square> legalMoves = new ArrayList<Square>();

        int column = this.getSquare().getCol();
        int rows = this.getSquare().getRow();
        Game gam = this.getSquare().getGame();

        // searching moves down from current location
        for (int i = rows + 1; i <= 9; i++) {
            Square newSquare = gam.getSquare(i, column);
            if (newSquare.canBeEntered()) {

               legalMoves.add(newSquare);

            } else {

                break;
            }
        }

        // searching moves UP from current location
        for (int i = rows - 1; i >= 0; i--) {
            Square newSquare = gam.getSquare(i, column);
            if (newSquare.canBeEntered()) {

                legalMoves.add(newSquare);

            } else {

                break;
            }
        }

        // searching moves RIGHT from current location
        for (int i = column + 1; i <= 9; i++) {
            Square newSquare = gam.getSquare(rows, i);
            if (newSquare.canBeEntered()) {

                legalMoves.add(newSquare);

            } else {

                break;
            }
        }
        // searching moves LEFT from current location
        for (int i = column - 1; i >= 0; i--) {
            Square newSquare = gam.getSquare(rows, i);
            if (newSquare.canBeEntered()) {

                legalMoves.add(newSquare);

            } else {

                break;
            }
        }

        return legalMoves;
    }

}

