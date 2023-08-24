import java.util.List;

public abstract class SweepApproach {
    KBase kBase=null;
    private boolean probeCell=false;

    public SweepApproach(){
    }

    public void setKBase(KBase kBase) {
        this.kBase = kBase;
    }

    public KBase getKBase() {
        return kBase;
    }

    public abstract List<Cell> getNextProbe();

    public boolean isOKtoProbeCell() {
        return probeCell;
    }

    public void setProbeCell(boolean yesNo) {
        this.probeCell = yesNo;
    }

    public abstract List<Cell> getFirstMove();
}
