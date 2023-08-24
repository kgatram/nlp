import java.util.ArrayList;
import java.util.List;

/**
 * Moves enum.
 * This enum contains all the possible moves ferry can make.
 * It also contains a method to get all the valid moves for a state.
 *
 * @author 220025456
 */
public enum Moves {
    /**
     * Move to right.
     */
    Right(0, 1),
    /**
     * Move to left.
     */
    Left(0, -1),
    /**
     * Move up.
     */
    Up(-1, 0),
    /**
     * Move down action.
     */
    Down(1, 0),
    /**
     * Move up and right action.
     */
    Up_Right(-1, 1),
    /**
     * Move up and left action.
     */
    Up_Left(-1, -1),
    /**
     * Move down and right action.
     */
    Down_Right(1, 1),
    /**
     * Move down and left action.
     */
    Down_Left(1, -1);

    private final int row;
    private final int col;

    /**
     * Constructor to move row and cols
     *
     * @param row move row
     * @param col move col
     */
    Moves(int row, int col) {
        this.row = row;
        this.col = col;
    }

    /**
     * Gets all valid moves for a coordinate.
     *
     * @param coord         current state
     * @param diagonalMoves option to allow advanced moves
     * @return valid moves that a state can use to transition into another state
     */
    public static ArrayList<Moves> getAllowedMoves(Coord coord, boolean diagonalMoves) {
        ArrayList<Moves> moves;
        boolean upTriangle = (coord.getR() + coord.getC()) % 2 == 0;
        if (upTriangle) {
            if (diagonalMoves) {
                moves = new ArrayList<>(List.of(Right, Down_Right, Down, Down_Left, Left));
            } else {
                moves = new ArrayList<>(List.of(Right, Down, Left));
            }
        } else {
            if (diagonalMoves) {
                moves = new ArrayList<>(List.of(Right, Left, Up_Left, Up, Up_Right));
            } else {
                //downward triangle
                moves = new ArrayList<>(List.of(Right, Left, Up));
            }
        }
        return moves;
    }

    /**
     * move to successor coordinates
     *
     * @param coord current coordinates
     * @return new coordinates
     */
    public Coord move(Coord coord) {
        return new Coord(coord.getR() + row, coord.getC() + col, this.name());
    }


}
