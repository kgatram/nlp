public enum Status {
    TORNADO("Result: Agent dead: found mine\n"),
    WON("Result: Agent alive: all solved\n"),
    NO_MOVES("Result: Agent not terminated\n"),
    GAME_ON("Result: Running");
    private final String message;

    Status(String message) {
        this.message = message;
    }

    public String getMessage() {
        return message;
    }

}
