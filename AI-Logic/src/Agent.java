import java.util.List;

/**
 *  Intelligent Agent playing the tornado sweeper game
 * @author 220025456
 */

public class Agent {
    private Board board;
    private KBase kBase;
    private SweepApproach approach;
    private final boolean verbose;
    private boolean firstMove = false;

    public Agent(boolean verbose) {
        this.verbose = verbose;
    }

    public void setKBase(KBase kBase) {
        this.kBase = kBase;
    }

    public void setBoard(Board board) {
        this.board = board;
    }

    public void setApproach(SweepApproach approach) {
        this.approach = approach;
    }

    public Board getBoard() {
        return board;
    }

    private void initialize() {
        // add all the pit to the knowledge base
        board.getPitCells().forEach(this::uncoverCell);
        if (verbose) {
            logger();
        }
    }

    public void play() {
        initialize();
        while (true) {
            // do probing
            List<Cell> checkCells = getNextProbe();

            if (!checkCells.isEmpty()) {
                checkCells.forEach(cell -> {
                    if (board.getStartCells().contains(cell) || approach.isOKtoProbeCell()) {
                        probe(cell);
                    } else {
                        flagCell(cell);
                    }
                });
            } else {
                // no more logical moves
                board.noMoves();
                break;
            }

            if (!board.isGameON(approach)) {
                break;
            } else {
                if (verbose) {
                    logger();
                }
            }
        }
        gameOver();
    }


    private List<Cell> getNextProbe() {
        if (firstMove) {
            return approach.getNextProbe();
        } else {
            firstMove = true;
            return approach.getFirstMove();
        }
    }

    private void flagCell(Cell cell) {
        kBase.flagCell(cell);
    }

    private void probe(Cell cell) {
        if (kBase.getUncoveredCells().contains(cell)) {
            return;
        }

        CellType type = getBoard().probe(cell);
        if (type == CellType.TORNADO) {
            kBase.uncoverCell(cell, '-');
        } else {
            uncoverCell(cell);
        }

    }

    private void uncoverCell(Cell cell) {
        CellType type = getBoard().probe(cell);
        kBase.uncoverCell(cell, type.getCharValue());
        // Uncover all neighbours if current cell clue is zero.
        if (type == CellType.CELL0) {
            kBase.getHiddenNeighbours(cell).forEach(this::uncoverCell);
        }

    }

    private void gameOver() {
        System.out.println("Final map");
        logger();
        board.gameOver();
    }


    public static void printBoard(char[][] board) {
        System.out.println();
        // first line
        System.out.print("    ");
        for (int j = 0; j < board[0].length; j++) {
            System.out.print(j + " "); // x indexes
        }
        System.out.println();
        // second line
        System.out.print("    ");
        for (int j = 0; j < board[0].length; j++) {
            System.out.print("- ");// separator
        }
        System.out.println();
        // the board
        for (int i = 0; i < board.length; i++) {
            System.out.print(" " + i + "| ");// index+separator
            for (int j = 0; j < board[0].length; j++) {
                System.out.print(board[i][j] + " ");// value in the board
            }
            System.out.println();
        }
        System.out.println();
    }

    private void logger() {
        if (board.isSquareBoard()) {
            printBoard(kBase.getBoardView());
        } else {
            A3main.printBoard(kBase.getBoardView());
        }
    }

    public void logger(char[][] map) {
        if (board.isSquareBoard()) {
            printBoard(map);
        } else {
            A3main.printBoard(map);
        }
    }

}
