/**
 * This class converts row and column coordinates to triangle a, b, c coordinates.
 *
 * @author 220025456
 */

public class TriangleCoord {
    private final int a;
    private final int b;
    private final int c;

    /**
     * Constructor
     *
     * @param a A
     * @param b B
     * @param c C
     */
    public TriangleCoord(int a, int b, int c) {
        this.a = a;
        this.b = b;
        this.c = c;
    }

    /**
     * Convert row col to  triangle coordinate.
     *
     * @param rc coordinate as row and column
     * @return abc triangle coordinate
     */
    public static TriangleCoord convertRowCol(Coord rc) {
        int dir = (rc.getR() + rc.getC()) % 2 == 0 ? 0 : 1;
        int row = rc.getR();
        int col = rc.getC();
        int a = -row;
        int b = (row + col - dir) / 2;
        int c = b - row + dir;
        return new TriangleCoord(a, b, c);
    }

    /**
     * @return get A
     */
    public int getA() {
        return a;
    }

    /**
     * @return get B
     */
    public int getB() {
        return b;
    }

    /**
     * @return get C
     */
    public int getC() {
        return c;
    }

}
