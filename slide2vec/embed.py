def main(*_args, **_kwargs):
    raise SystemExit(
        "slide2vec.embed is no longer a standalone entrypoint. "
        "Use `python -m slide2vec` or the `slide2vec.Model` / `slide2vec.Pipeline` API."
    )


if __name__ == "__main__":
    main()
