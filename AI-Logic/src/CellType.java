import java.util.Arrays;

public enum CellType {
    COVERED('?',-3),
    TORNADO('t', -2),
    PIT('p', -1),
    CELL0('0', 0),
    CELL1('1', 1),
    CELL2('2', 2),
    CELL3('3', 3),
    CELL4('4', 4),
    CELL5('5', 5),
    CELL6('6', 6),
    CELL7('7', 7),
    CELL8('8', 8);


    private final char charValue;
    private final int intValue;

    CellType(char charValue, int intValue) {

        this.charValue = charValue;
        this.intValue = intValue;
    }

    public char getCharValue() {
        return charValue;
    }

    public int getIntValue() {
        return intValue;
    }

    public static CellType resolve(char c) {
        return Arrays.stream(CellType.values()).filter(ct -> ct.getCharValue() == c).findFirst().orElse(null);
    }


}
