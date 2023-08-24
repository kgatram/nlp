import java.util.ArrayList;
import java.util.List;

/**
 * Board class to represent the game layout
 * @author 220025456
 */
public class Board {
    private final char[][] map;
    private final boolean squareBoard;
    private final int boardWidth;
    private final int boardHeight;
    private final int numberOfMines;
    private Status status;


    public Board(char[][] map, boolean square) {
        this.map = map;
        this.squareBoard = square;
        numberOfMines = countMines();
        status = Status.GAME_ON;
        boardHeight = map.length;
        boardWidth = map[0].length;
    }


    public int getBoardWidth() {
        return boardWidth;
    }

    public int getBoardHeight() {
        return boardHeight;
    }

    public boolean isSquareBoard() {
        return squareBoard;
    }

    public List<Cell> getStartCells() {
        List<Cell> firstMove = new ArrayList<>();
        firstMove.add(new Cell(0, 0));
        firstMove.add(new Cell((getBoardHeight() - 1) / 2, (getBoardWidth() - 1) / 2));
        return firstMove;
    }

    public boolean checkWin(SweepApproach approach) {
        boolean hasWon;
        if (approach instanceof LexicOrder) {
            hasWon = (boardWidth * boardHeight) - (approach.getKBase().getUncoveredCells().size() + countMines()) == 0;

        } else {
            hasWon = approach.getKBase().getCoveredCells().size() == 0;
        }
        if (hasWon && status != Status.TORNADO) {
            status = Status.WON;
        }
        return hasWon;
    }

    public boolean isGameON(SweepApproach approach) {

        return !checkWin(approach) && status == Status.GAME_ON;
    }

    private int countMines() {
        int count = 0;
        for (char[] rows : map) {
            for (char type : rows) {
                if (CellType.resolve(type) == CellType.TORNADO) {
                    count++;
                }
            }
        }
        return count;
    }


    public CellType probe(Cell cell) {
        CellType cellType = CellType.resolve(map[cell.getR()][cell.getC()]);
        if (cellType == CellType.TORNADO) {
            status = Status.TORNADO;
        }
        return cellType;
    }


    public void noMoves() {
        status = Status.NO_MOVES;
    }

    public void gameOver() {
        printResult();
    }

    public void printResult() {
        System.out.println("\n" + status.getMessage());
    }

    public List<Cell> getPitCells() {
        List<Cell> cells = new ArrayList<>();
        for (int i = 0; i < map.length; i++) {
            for (int j = 0; j < map[0].length; j++) {
                CellType cellType = CellType.resolve(map[i][j]);
                if (cellType == CellType.PIT) {
                    cells.add(new Cell(i, j));
                }
            }
        }
        return cells;
    }

}
