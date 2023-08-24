
/**
 * Main class to run the game.
 * Usage: java A3main <P1|P2|P3|P4> <world> [verbose] [square]
 * @author 220025456
 */

public class A3main {

    public static void main(String[] args) {

        try {
            System.out.println("-------------------------------------------\n");
            System.out.println("Agent " + args[0] + " plays " + args[1] + "\n");

            boolean verbose= args.length > 2 && args[2].equalsIgnoreCase("verbose");
            boolean square= args.length > 3 && args[3].equalsIgnoreCase("square");
            World world = World.valueOf(args[1]);
            SweepApproach approach = getApproach(args[0]);
            gameSetup(world, approach, verbose, square);
        } catch (ArrayIndexOutOfBoundsException aioobe) {
            System.out.println("Usage: java A3main <P1|P2|P3|P4> <world> [verbose] [square]");
        }

    }


    //prints the board in the required format - PLEASE DO NOT MODIFY
    public static void printBoard(char[][] board) {
        System.out.println();
        // first line
        for (int l = 0; l < board.length + 5; l++) {
            System.out.print(" ");// shift to start
        }
        for (int j = 0; j < board[0].length; j++) {
            System.out.print(j);// x indexes
            if (j < 10) {
                System.out.print(" ");
            }
        }
        System.out.println();
        // second line
        for (int l = 0; l < board.length + 3; l++) {
            System.out.print(" ");
        }
        for (int j = 0; j < board[0].length; j++) {
            System.out.print(" -");// separator
        }
        System.out.println();
        // the board
        for (int i = 0; i < board.length; i++) {
            for (int l = i; l < board.length - 1; l++) {
                System.out.print(" ");// fill with left-hand spaces
            }
            if (i < 10) {
                System.out.print(" ");
            }

            System.out.print(i + "/ ");// index+separator
            for (int j = 0; j < board[0].length; j++) {
                System.out.print(board[i][j] + " ");// value in the board
            }
            System.out.println();
        }
        System.out.println();
    }

    private static SweepApproach getApproach(String inString) {
        SweepApproach approach;
        switch (inString) {
            case "P1":
            default:
                //Part 1: Lexicographic ordering
                approach = new LexicOrder();
                break;
            case "P2":
                //Part 2: Single Point Strategy
                approach = new SPS();
                break;
            case "P3":
                //TODO: Part 3
                approach = new DNF();
                break;
            case "P4":
                //TODO: Part 4
                approach = new CNF();
                break;
        }
        return approach;
    }

    private static void gameSetup(World world, SweepApproach approach, boolean verbose, boolean square) {
        char[][] map = world.getMap();

        Board board = new Board(map, square);
        KBase kBase = new KBase(board.getBoardWidth(), board.getBoardHeight(), square);
        approach.setKBase(kBase);
        Agent agent = new Agent(verbose);
        agent.setKBase(kBase);
        agent.setApproach(approach);
        agent.setBoard(board);

//        printBoard(map);
        agent.logger(map);
        System.out.println("Start!");
        agent.play();
    }

}