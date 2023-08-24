import java.util.List;

/**
 * Single Point Strategy - Scan all cells one by one. For each cell that is covered check its adjacent neighbours
 * If the cell has:
 * All Free Neighbours(AFN): uncover
 * All Marked Neighbours (AMN): flag a tornado
 *
 * @author 220025456
 */
public class SPS extends SweepApproach{
    /**
     * To get the next probe, get all the uncovered neighbours of covered cell
     * and check if any of uncovered neighbours satisfy afn or amn.
     * If afn is satisfied then return the covered cell to probe.
     * If amn is satisfied then return the covered cell to flag (mark) as tornado.
     * @return List of Cells
     */
    public List<Cell> getNextProbe() {

            for (Cell coveredCell : kBase.getCoveredCells()) {
                for (Cell uncoveredNeighbour : kBase.getUncoveredNeighbours(coveredCell)) {
                    if (kBase.getHiddenNeighbours(uncoveredNeighbour).size() > 0) {
                        if (afn(uncoveredNeighbour)) {
                            setProbeCell(true);
                            return List.of(coveredCell);
                        } else if (amn(uncoveredNeighbour)) {
                            setProbeCell(false);
                            return List.of(coveredCell);
                        }
                    }
                }
            }
            return List.of();
        }

    /**
     * All Free Neighbours
     * If the uncovered cell value and the number of flagged neighbours are equal then afn is satisfied.
     * @param cell uncovered cell
     * @return true if afn is satisfied
     */
    private boolean afn(Cell cell) {
            int clue = kBase.getCellType(cell).getIntValue();
            int flaggedNeighbours = kBase.getFlaggedNeighbours(cell).size();
            return flaggedNeighbours == clue;
        }

    /**
     * All Marked Neighbours
     * If the uncovered cell value - the number of it's flagged neighbours is equal
     * to its uncovered neighbours then amn is satisfied.
     * @param cell uncovered cell
     * @return true if amn is satisfied
     */
    private boolean amn(Cell cell) {
        int clue = kBase.getCellType(cell).getIntValue();
        int flaggedNeighbours = kBase.getFlaggedNeighbours(cell).size();
        return kBase.getHiddenNeighbours(cell).size() == (clue - flaggedNeighbours);
    }

        @Override
    public List<Cell> getFirstMove() {
        return List.of(
                new Cell(0, 0),
                new Cell(kBase.getHeight() / 2, kBase.getWidth() / 2)
        );
    }
}
