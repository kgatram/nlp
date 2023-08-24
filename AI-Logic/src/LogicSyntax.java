import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Class to convert to logic sentences
 * @author 220025456
 */

public class LogicSyntax {

    public static String OR_SYMBOL = "|";
    public static String AND_SYMBOL = "&";
    public static String NOT_SYMBOL = "~";

    public static String toLiteral(Cell cell) {
        return "T_" + cell.getR() + "_" + cell.getC();
    }

    public static Cell toCell(String literal) {
        String[] coords = literal.replace("T_", "").split("_");
        return new Cell(Integer.parseInt(coords[0]), Integer.parseInt(coords[1]));
    }


    public static String orAll(List<String> clauses) {
        return group(clauses.stream().collect(Collectors.joining(OR_SYMBOL)));
    }


    public static String andAll(List<String> clauses) {
        return group(clauses.stream().collect(Collectors.joining(AND_SYMBOL)));
    }

    public static String not(String a) {
        return NOT_SYMBOL + a;
    }

    public static int not(int a) {
        return -a;
    }

    public static String group(String a) {
        return "(" + a + ")";
    }

    public static List<int[]> combinator(int n, int r) {
        if (r == 0) {
            return List.of(new int[]{});
        }
        List<int[]> combinations = new ArrayList<>();
        int[] combination = new int[r];

        for (int i = 0; i < r; i++) {
            combination[i] = i;
        }

        while (combination[r - 1] < n) {
            combinations.add(combination.clone());

            int t = r - 1;
            while (t != 0 && combination[t] == n - r + t) {
                t--;
            }
            combination[t]++;
            for (int i = t + 1; i < r; i++) {
                combination[i] = combination[i - 1] + 1;
            }
        }

        return combinations;
    }

}
