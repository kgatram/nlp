import java.util.*;
import java.util.stream.Collectors;

import org.logicng.datastructures.Tristate;
import org.logicng.formulas.Formula;
import org.logicng.formulas.FormulaFactory;
import org.logicng.io.parsers.ParserException;
import org.logicng.io.parsers.PropositionalParser;
import org.logicng.solvers.MiniSat;
import org.logicng.solvers.SATSolver;

/**
 * DNF using LogicNG
 * @author 220025456
 */

public class DNF extends SPS{

    public List<Cell> getNextProbe() {
        // get next probe
        Iterator<Cell> cellIterator = kBase.getHiddenCellIterator();
        while (cellIterator.hasNext()) {
            Cell cell = solve(cellIterator.next());
            if (cell != null) {
                return List.of(cell);
            }
        }
        // dnf could not resolve any cell try sps
        return super.getNextProbe();
    }


    /**
     * We can prove that the cell is clear in [x, y]
     * KB |= ¬Dx,y invoke satsolver(“KB ∧ Dx,y ”)
     * If false the cell is clear – Probe!
     * @param cell
     * @return returns cell to probe.
     */

    public Cell solve(Cell cell) {
        // get current cell
        Set<String> kbu = buildKBU();
        // add current cell, to check if the cell is a tornado
        kbu.add(LogicSyntax.toLiteral(cell));
        FormulaFactory f = new FormulaFactory();
        PropositionalParser p = new PropositionalParser(f);
        SATSolver miniSat = MiniSat.miniSat(f);
        for (String sentence : kbu) {
            try {
                Formula formula = p.parse(sentence);
                miniSat.add(formula);
            } catch (ParserException e) {
                e.printStackTrace();
            }
        }

        // test satisfiability
        Tristate result = miniSat.sat();
        if (result != Tristate.TRUE) {
            setProbeCell(true);
            return cell;
        }
        return null;
    }


    /**
     * Build the knowledge base of unknown using DNF encoding.
     * @return logic DNF sentences
     */
    private Set<String> buildKBU() {
        Set<String> sentences = new HashSet<>();
        kBase.getFlaggedCells().forEach(cell -> sentences.add(LogicSyntax.toLiteral(cell)));
        List<String> uncoveredCellKBU = new ArrayList<>();

        kBase.getUncoveredCells().forEach(cell -> {
            CellType type = kBase.getCellType(cell);
            List<Cell> thisCellNeighbours = kBase.getHiddenNeighbours(cell);

            if (type != CellType.PIT && thisCellNeighbours.size() > 0) {

                int clue = type.getIntValue();
                clue  = clue - kBase.getFlaggedNeighbours(cell).size();

                List<int[]> combinations = LogicSyntax.combinator(thisCellNeighbours.size(), clue);
                List<String> clauses = new ArrayList<>();
                combinations.forEach(combination->{

                    String[] arr = thisCellNeighbours.stream().map(LogicSyntax::toLiteral).toArray(String[]::new);
                    List<Integer> combs = Arrays.stream(combination).boxed().toList();

                    for (int i = 0; i < arr.length; i++) {
                        if(!combs.contains(i)){
                            arr[i]  = LogicSyntax.not(arr[i]); // negate the cell ~T_1_0
                        }
                    }
                    // "(T_0_1&~T_1_0)"
                    clauses.add(LogicSyntax.andAll(Arrays.stream(arr).collect(Collectors.toList())));
                });
                // "((T_0_1&~T_1_0)|(~T_0_1&T_1_0))"
                uncoveredCellKBU.add(LogicSyntax.orAll(clauses));
            }
        });

        if (!uncoveredCellKBU.isEmpty()) {
            sentences.add(LogicSyntax.andAll(uncoveredCellKBU));
        }
        return sentences;
    }

}
