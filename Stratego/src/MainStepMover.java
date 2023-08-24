import stratego.*;
import stratego.pieces.*;

import java.util.List;

public class MainStepMover {
    public static void main(String[] args) {

        Player p0;
        Player p1;
        Game game;
        Piece marshal, sergeant, captain;

        p0 = new Player("Michael", 0);
        p1 = new Player("Ozgur", 1);
        game = new Game(p0, p1);
        marshal = new StepMover(p0, game.getSquare(0, 3), 10);
        sergeant = new StepMover(p1, game.getSquare(0, 2), 4);
        captain = new StepMover(p1, game.getSquare(1, 2), 6);

        System.out.println("Marshal when attacking Sergeant = " + marshal.resultWhenAttacking(sergeant));

        List<Square> moves = marshal.getLegalMoves();
        System.out.println("Move size = " + moves.size());
        System.out.println(moves.contains(game.getSquare(0, 4)));
        System.out.println(moves.contains(game.getSquare(1, 3)));
        System.out.println(moves.contains(game.getSquare(0, 2)));

        List<Square> attacks = marshal.getLegalAttacks();
        System.out.println("Attack size = " + attacks.size());
        System.out.println(attacks.contains(game.getSquare(0, 2)));
        System.out.println(attacks.contains(game.getSquare(1, 2)));

    }
}
