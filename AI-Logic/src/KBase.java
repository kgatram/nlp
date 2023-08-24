import java.util.*;
import java.util.stream.Collectors;

/**
 * Knowledge base for the tornado world.
 * @author 220025456
 */
public class KBase {
    private final char[][] boardView;
    private final boolean squareBoard;
    private final int Height;
    private final int Width;
    private final Set<Cell> coveredCells;
    private final Set<Cell> uncoveredCells;
    private final Set<Cell> flaggedCells;

    public KBase(int Width, int Height, boolean squareBoard) {
        // linked list is used to maintain order of insertion.
        this.Height = Height;
        this.Width = Width;
        this.squareBoard = squareBoard;
        boardView = new char[this.Height][this.Width];
        flaggedCells = new LinkedHashSet<>();
        coveredCells = new LinkedHashSet<>();
        uncoveredCells = new LinkedHashSet<>();
        for (int i = 0; i < this.Height; i++) {
            for (int j = 0; j < this.Width; j++) {
                boardView[i][j] = CellType.COVERED.getCharValue();
                coveredCells.add(new Cell(i, j));
            }
        }
    }

    public int getHeight() {
        return Height;
    }

    public int getWidth() {
        return Width;
    }

    public char[][] getBoardView() {
        return boardView;
    }

    public Set<Cell> getCoveredCells() {
        return coveredCells;
    }

    public Set<Cell> getUncoveredCells() {
        return uncoveredCells;
    }

    public void uncoverCell(Cell cell, char value) {
        if (coveredCells.contains(cell)) {
            boardView[cell.getR()][cell.getC()] = value;
            coveredCells.remove(cell);
            uncoveredCells.add(cell);
        }

    }

    public void flagCell(Cell cell) {
        // remove cell from hidden and add it to flagged
        if (coveredCells.contains(cell)) {
            boardView[cell.getR()][cell.getC()] = '*';
            coveredCells.remove(cell);
            flaggedCells.add(cell);
        }

    }

    public CellType getCellType(Cell cell) {
        return CellType.resolve(boardView[cell.getR()][cell.getC()]);
    }

    public Cell getNextCoveredCell() {
        if (this.getCoveredCells().size() != 0) {
            Iterator<Cell> it = this.getCoveredCells().iterator();
            return it.next();
        }
        return null;
    }

    public Iterator<Cell> getHiddenCellIterator() {
        return this.getCoveredCells().iterator();
    }

    public List<Cell> getNeighbours(Cell cell) {

        int r = cell.getR();
        int c = cell.getC();
        List<Cell> neighbours = new ArrayList<>();
        for (int i = -1; i < 2; i++) {
            for (int j = -1; j < 2; j++) {
                Cell neighbour = new Cell(r + i, c + j);

                if (squareBoard) {
                    if (!neighbour.equals(cell) && isOnBoard(neighbour)) {
                        neighbours.add(neighbour);
                    }
                } else {
//                in case of hexagonal board i=-1 and j=1 is not a neighbour, similarly i=1 and j=-1.
                    if (!(i*j == -1) && !neighbour.equals(cell) && isOnBoard(neighbour)) {
                        neighbours.add(neighbour);
                    }
                }
            }
        }
        return neighbours;
    }

    public List<Cell> getHiddenNeighbours(Cell cell) {
        return getNeighbours(cell).stream().filter(coveredCells::contains).collect(Collectors.toList());
    }

    public List<Cell> getFlaggedNeighbours(Cell cell) {
        return getNeighbours(cell).stream().filter(flaggedCells::contains).collect(Collectors.toList());
    }

    public List<Cell> getUncoveredNeighbours(Cell cell) {
        return getNeighbours(cell).stream().filter(uncoveredCells::contains).collect(Collectors.toList());
    }

    public Set<Cell> getFlaggedCells() {
        return flaggedCells;
    }

    public boolean isOnBoard(Cell cell) {
        return cell.getR() >= 0 && cell.getR() < boardView.length && cell.getC() >= 0 && cell.getC() < boardView[0].length;
    }

}
