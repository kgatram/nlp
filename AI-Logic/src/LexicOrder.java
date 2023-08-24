import java.util.List;

public class LexicOrder extends SweepApproach {

    public List<Cell> getNextProbe() {
        Cell nextProbe = kBase.getNextCoveredCell();
        setProbeCell(true);
        return List.of(nextProbe);
    }

    @Override
    public List<Cell> getFirstMove() {
        return List.of(new Cell(0,0));
    }


}
