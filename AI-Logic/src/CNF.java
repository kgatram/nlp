import org.sat4j.core.VecInt;
import org.sat4j.minisat.SolverFactory;
import org.sat4j.specs.ContradictionException;
import org.sat4j.specs.IProblem;
import org.sat4j.specs.ISolver;
import org.sat4j.specs.TimeoutException;

import java.util.*;

/**
 * CNF using SAT4J
 * @author 220025456
 */
public class CNF extends SPS {

    private final int MAXVAR = 1000000;

    @Override
    public List<Cell> getNextProbe() {
        // get next probe
        Iterator<Cell> cellIterator = kBase.getHiddenCellIterator();
        while (cellIterator.hasNext()) {
            Cell cell = solve(cellIterator.next());
            if (cell != null) {
                return List.of(cell);
            }
        }
        // if not solved try sps
        return super.getNextProbe();
    }

    public Cell solve(Cell cell){
        Set<String> kbu = buildKBU();
        // add current cell
        kbu.add(LogicSyntax.toLiteral(cell));
        List<int[]> dimacs = convertToDimacs(kbu);
        ISolver solver = SolverFactory.newDefault();
        solver.newVar(MAXVAR);
        solver.setExpectedNumberOfClauses(dimacs.size());
        for (int[] dimac : dimacs) {
            try {
                solver.addClause(new VecInt(dimac));
            } catch (ContradictionException ignored) {
            }
        }
        IProblem problem = solver;
        try {
            if (!problem.isSatisfiable()) {
                setProbeCell(true);
                return cell;
            }
        } catch (TimeoutException e) {
            e.printStackTrace();
        }
        return null;
    }


    public Set<String> buildKBU() {
        Set<String> kbu = new HashSet<>();
        kBase.getUncoveredCells().forEach(cell -> {
            CellType type = kBase.getCellType(cell);
            List<Cell> cellHiddenNeighbours = kBase.getHiddenNeighbours(cell);
            if (type != CellType.PIT && cellHiddenNeighbours.size() > 0) {
                int clue = type.getIntValue();
                clue = clue - kBase.getFlaggedNeighbours(cell).size();
                kbu.addAll(xactly(cellHiddenNeighbours, clue));
            }
        });
        return kbu;
    }

    public List<String> xactly(List<Cell> cells, int clue) {
        List<String> xactlyClauses = new ArrayList<>();
        xactlyClauses.addAll(atMost(cells, clue, true));
        xactlyClauses.addAll(atMost(cells, cells.size() - clue, false));
        return xactlyClauses;
    }

    public List<String> atMost(List<Cell> neighbours, int clue, boolean isTornado) {
        List<int[]> combinations = LogicSyntax.combinator(neighbours.size(), clue + 1);
        List<String> clauses = new ArrayList<>();
        combinations.forEach(combination -> {
            List<String> literals = new ArrayList<>();
            for (int i = 0; i < combination.length; i++) {
                String literal = LogicSyntax.toLiteral(neighbours.get(combination[i]));
                if (isTornado) {
                    literal = LogicSyntax.not(literal);
                }
                literals.add(literal);
            }
            clauses.add(String.join(LogicSyntax.OR_SYMBOL, literals));
        });
        return clauses;
    }

    public List<int[]> convertToDimacs(Set<String> clauses) {
        List<int[]> dimacsClauses = new ArrayList<>();
        for (String clause : clauses) {

            List<Integer> literals = new ArrayList<>();
            for (String literal : clause.split("\\" + LogicSyntax.OR_SYMBOL)) {
                int multiplier = literal.contains(LogicSyntax.NOT_SYMBOL) ? -1 : 1;
                Cell cell = LogicSyntax.toCell(literal.replace(LogicSyntax.NOT_SYMBOL, ""));
                literals.add(cellToClauseId(cell) * multiplier);
            }
            dimacsClauses.add(literals.stream().mapToInt(i -> i).toArray());
        }
        return dimacsClauses;
    }

    public int cellToClauseId(Cell cell) {
        return (kBase.getWidth() * cell.getR()) + (1 + cell.getC());
    }

}
