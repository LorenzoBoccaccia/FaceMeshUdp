# Agent Rules

- Do not apply hacks to compensate math issues in gaze/pose pipelines.
- Debug-only behavior must not become runtime business logic unless explicitly requested and documented.
- Fix bugs at the source, not by patching the effects.
- Remove porkarounds don't build on top of them. 
- We need solid foundation, not layers of fixes. Make it correct from the origin onward instead of fixing errors in place.

python in .venv
test with pytest under test folder
docs under docs folder