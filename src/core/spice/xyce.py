import os
import shutil
import subprocess  # nosec B404
import tempfile

from PySpice.Spice.Xyce.RawFile import RawFile


class XyceSim:
    """Xyce simulator class."""

    def __init__(self, **kwargs):
        self._xyce_command = kwargs.get("xyce_command") or "Xyce"
        self._mpi_command = kwargs.get("mpi_command") or ["mpirun", "-use-hwthread-cpus"]
        assert type(self._mpi_command) is list, ValueError("mpi_command must be a list")

    def __call__(self, spice_input):
        """Run Xyce simulation and return RawFile instance."""
        tmp_dir = tempfile.mkdtemp()
        input_filename = os.path.join(tmp_dir, "input.cir")
        output_filename = os.path.join(tmp_dir, "output.raw")

        # input_filename = 'input.cir'
        # output_filename = 'output.raw'
        with open(input_filename, "w") as f:
            f.write(str(spice_input))
        """
        # print(mpi_command)
        command = self._mpi_command + [self._xyce_command, '-r', output_filename, input_filename]
        # print(command)
        # self._logger.info('Run {}'.format(' '.join(command)))
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        stdout, stderr = process.communicate()
        # self._parse_stdout(stdout)
        """
        command = [self._xyce_command, "-r", output_filename, input_filename]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # nosec B603

        with open(output_filename, "rb") as f:
            output = f.read()
        # self._logger.debug(output)

        raw_file = RawFile(output)
        shutil.rmtree(tmp_dir)

        return raw_file
