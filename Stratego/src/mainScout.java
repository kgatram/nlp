import stratego.*;
import stratego.pieces.*;

import java.util.List;

public class mainScout {
    public static void main(String[] args) {
        Player p0;
        Player p1;
        Game game;
        Piece marshal, sergeant, flag, scout;

        p0 = new Player("Michael", 0);
        p1 = new Player("Ozgur", 1);
        game = new Game(p0, p1);
        marshal = new StepMover(p0, game.getSquare(1, 4), 10);
        sergeant = new StepMover(p0, game.getSquare(0, 2), 4);
        flag = new Flag(p0, game.getSquare(2, 5));
        scout = new Scout(p1, game.getSquare(5, 4));

        System.out.println("Scout when attacking Sergeant = " + scout.resultWhenAttacking(sergeant));
        System.out.println("Scout when attacking marshal = " + scout.resultWhenAttacking(marshal));
        System.out.println("Scout when attacking Flag = " + scout.resultWhenAttacking(flag));
        System.out.println("Scout rank" + scout.getRank());

        List<Square> moves = scout.getLegalMoves();

        System.out.println("Move size = " + moves.size());
        System.out.println(moves.contains(game.getSquare(2, 4)));  // can move next to marshal
        System.out.println(moves.contains(game.getSquare(1, 4)));  // cannot move onto marshal
        System.out.println(moves.contains(game.getSquare(0, 4)));  // cannot jump over marshal
        System.out.println(moves.contains(game.getSquare(5, 5)));  // can move right onto land
        System.out.println(moves.contains(game.getSquare(5, 3)));  // cannot move left onto water
        System.out.println(moves.contains(game.getSquare(6, 5)));  // cannot move diagonally

        List<Square> attacks = scout.getLegalAttacks();
        System.out.println("Attack size = " + attacks.size());

        scout.move(game.getSquare(2, 4));  // move next to marshal
        attacks = scout.getLegalAttacks();
        System.out.println(attacks.contains(marshal.getSquare()));
//        scout.attack(marshal.getSquare());

        if (scout.getSquare() == null) {
            System.out.println("Scout is null" );
        }
        if (marshal.getSquare() != null) {
            System.out.println("Marshall not null" );
        }

        scout.move(game.getSquare(2, 4));  // move next to flag
        attacks = scout.getLegalAttacks();
        System.out.println(attacks.contains(flag.getSquare()));
        scout.attack(flag.getSquare());
        if (scout.getSquare() == game.getSquare(2, 5)) {
            System.out.println("Scout is not null" );
        }
        System.out.println(game.getWinner().getName());
    }

}
