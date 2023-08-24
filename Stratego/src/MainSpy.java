import stratego.*;
import stratego.pieces.*;

public class MainSpy {
    public static void main(String[] args) {

        Player p0;
        Player p1;
        Game game;
        Piece marshal, sergeant, spy;

        p0 = new Player("Michael", 0);
        p1 = new Player("Ozgur", 1);
        game = new Game(p0, p1);
        marshal = new StepMover(p0, game.getSquare(1, 4), 10);
        sergeant = new StepMover(p0, game.getSquare(2, 5), 4);
        spy = new Spy(p1, game.getSquare(2, 4));

        System.out.println("Spy when attacking Sergeant = " + spy.resultWhenAttacking(sergeant));
        System.out.println("Spy when attacking marshal = " + spy.resultWhenAttacking(marshal));
        System.out.println("Marshal when attacking Spy = " + marshal.resultWhenAttacking(spy));

    }
}
