# Mars Rover

Windows:

```powershell
.\scripts\setup.ps1
.\scripts\windows\play.ps1 -Debug -Seed (Get-Random)
.\scripts\windows\play.ps1 -RefreshBank -BiomeCount 8 -Debug -Seed (Get-Random)
```

Linux:

```bash
scripts/setup.sh Release
.venv/bin/mars-rover-play --debug
```
