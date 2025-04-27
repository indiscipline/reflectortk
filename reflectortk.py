#!/usr/bin/env python3

# ReflectorTK
# Copyright 2025 Indiscipline
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, see <https://www.gnu.org/licenses/>.

"""
ReflectorTK: A graphical interface for managing Arch Linux mirrors using
reflector.

This program provides a GUI to configure and run the `reflector` command-line
tool, simplifying the process of generating an optimized mirrorlist. It's a
direct Python replacement for the `reflector-simple` bash script requiring no
third-party dependencies for its operation.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import shutil
import subprocess
import re
import os
import sys
import tempfile
import configparser
import copy
import urllib.request
import urllib.error
import queue
import locale
import ipaddress
import logging
import shlex
from enum import Enum, auto
from collections import namedtuple, UserDict
from threading import Thread
from typing import Dict, List, Optional, Tuple, Any, Set
from string import whitespace

# Constants
APP_NAME = "ReflectorTK"
WINDOW_TITLE = f"{APP_NAME}: Simple Reflector GUI"
REFLECTOR_CMD = "reflector"
PKEXEC_CMD = "pkexec"

## Configuration and System Paths
REFLECTOR_CONF_PATH = "/etc/xdg/reflector/reflector.conf"
CONFIG_FILE_NAME = APP_NAME.lower() + ".ini"
MIRRORLIST_PATH = "/etc/pacman.d/mirrorlist"
BACKUP_MIRRORLIST_PATH = MIRRORLIST_PATH + ".bak"
ARCH_MIRRORLIST_URL = "https://www.archlinux.org/mirrorlist/all"  # not passed on main reflector invocation
SERVER_PREFIX = "Server = "
COMMENTED_SERVER_PREFIX = "#" + SERVER_PREFIX
DEFAULT_NET_TIMEOUT = 5
SORT_OPTIONS = ["age", "rate", "country", "score", "delay"]  # Valid --sort keys
PROTOCOLS = {"https", "http"}  # ignoring "rsync"

GuiConfig = Dict[str, Any]

DEFAULT_GUI_CONFIG: GuiConfig = {
    "columns": 6,  # for country checkbox frame
    "min_width": 800,
    "min_height": 600,
}


# Data structures
class MsgKind(Enum):
    Start = auto()
    Done = auto()
    Msg = auto()


ROptVal = namedtuple("ROptVal", field_names=["long", "short", "parseas", "default"])


class ROpt(Enum):
    """Enum for reflector command-line options."""

    AGE = ROptVal("--age", "-a", "float", 2.0)
    CACHE_TIMEOUT = ROptVal("--cache-timeout", None, "int", 300)
    COUNTRIES = ROptVal("--country", "-c", "set", set())
    CONNECTION_TIMEOUT = ROptVal("--connection-timeout", None, "int", 5)
    DOWNLOAD_TIMEOUT = ROptVal("--download-timeout", None, "int", 5)
    LATEST = ROptVal(
        "--latest", "-l", "int", None
    )  # Prioritize --number by default, mutually exclusive
    LIST_COUNTRIES = ROptVal("--list-countries", None, None, None)  # Not in config
    NUMBER = ROptVal("--number", "-n", "int", 20)
    PROTOCOLS = ROptVal("--protocol", "-p", "set", {"https"})
    SAVE = ROptVal("--save", None, "str", None)  # Ignore!
    SORT = ROptVal("--sort", None, "str", SORT_OPTIONS[1])
    THREADS = ROptVal("--threads", None, "int", 1)
    URL = ROptVal(
        "--url", None, "str", None
    )  # ARCH_MIRRORLIST_URL - relegate to Reflector
    VERBOSE = ROptVal("--verbose", None, "bool", False)
    # Not in reflector-simple
    DELAY = ROptVal("--delay", None, "int", None)
    FASTEST = ROptVal("--fastest", "-f", "int", None)
    INCLUDE = ROptVal("--include", "-i", "str", None)  # Regex string
    EXCLUDE = ROptVal("--exclude", "-x", "str", None)  # Regex string
    SCORE = ROptVal("--score", None, "int", None)
    COMPLETION_PERCENT = ROptVal("--completion-percent", None, "float", None)
    IPV4 = ROptVal("--ipv4", None, "bool", False)
    IPV6 = ROptVal("--ipv6", None, "bool", False)
    CUSTOM_ARGS = ROptVal("--custom", None, "str", "")  # Not passed to Reflector

    def __str__(self) -> str:
        return self.value.long


OPT_STRING_TO_ENUM: Dict[str, ROpt] = {
    string: opt
    for opt in ROpt
    for string in [opt.value.long] + ([opt.value.short] if opt.value.short else [])
}

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,  # Log to stderr by default
)


# Classes
class ProgressModal(tk.Toplevel):
    def __init__(self, master, title="Processing...", text="Please, wait..."):
        super().__init__(master)
        self.withdraw()  # Hide window until configured
        self.minsize(320, 64)
        self.title(title)
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", lambda: None)  # Prevent closing via 'X'

        # Status Label Variable
        self.status_var = tk.StringVar(value=text)

        # Widgets
        ttk.Label(self, textvariable=self.status_var, anchor="w").pack(
            pady=(10, 0), padx=8, fill=tk.X
        )
        self.progress = ttk.Progressbar(self, orient="horizontal", mode="indeterminate")
        self.progress.pack(pady=8, padx=8, fill=tk.X)
        # Modality and Positioning
        self.transient(master)  # Keep on top of master
        self.grab_set()  # Make modal
        self.deiconify()  # Show window
        self.update_idletasks()  # Ensure window size is calculated
        # Center window
        x = master.winfo_x() + (master.winfo_width() - self.winfo_reqwidth()) // 2
        y = master.winfo_y() + (master.winfo_height() - self.winfo_reqheight()) // 2
        self.geometry(f"+{x}+{y}")
        self.progress.start(60)  # Start animation

    def update_status(self, message: str):
        """Updates the status label text (thread-safe)."""
        self.after(0, lambda: self.status_var.set(message))

    def stop(self):
        """Stops the progress bar and closes the window."""
        if self.winfo_exists():
            self.progress.stop()
            self.grab_release()
            self.destroy()


class SpinningMailbox:
    def _check_mailbox(self):
        try:
            msg, val = self.chan.get(block=False)
            match msg:
                case MsgKind.Start:
                    if self.running and self._progress is not None:
                        self._progress.stop()
                    self.running = True
                    title, text = val
                    self._progress = ProgressModal(self.master, title, text)
                case MsgKind.Done:
                    if self._progress is not None:
                        self._progress.stop()
                    self.running = False
                    self._progress = None
                case MsgKind.Msg:
                    if self._progress is not None:
                        self._progress.update_status(val)
        except queue.Empty:
            pass
        self.master.update_idletasks()
        if self.running:
            self.master.after(50, self._check_mailbox)

    def __init__(self, master):
        self.running = False
        self._progress = None
        self.master = master
        self.chan = queue.SimpleQueue()

    def start(self):
        self.running = True
        self.master.after(50, self._check_mailbox)


class ReflectorConfig(UserDict[ROpt, Any]):
    def __setitem__(self, key, item):
        if not isinstance(key, ROpt):
            raise KeyError(key)
        self.data[key] = item

    @classmethod
    def default(cls):
        return cls({k: k.value.default for k in ROpt if k.value.default is not None})

    def toggle_set_el(self, par: ROpt, el: str, b: bool):
        self[par].add(el) if b else self[par].discard(el)
        logging.debug(f"Toggled {par}: '{el}' to {b}.")

    def as_args(self):
        args = [ROpt.VERBOSE.value.long]
        for k, v in self.items():
            if v:
                if k == ROpt.CUSTOM_ARGS:
                    args.extend(shlex.split(v))
                    continue
                else:
                    args.append(k.value.long)
                    args.append(",".join(v) if isinstance(v, set) else str(v))
        # logging.debug(f"Generated reflector args: {args}")
        return args

    def as_cfg_dict(self) -> Dict[str, str]:
        cfg = {}
        for opt, value in self.items():
            ini_key = opt.value.long.lstrip("-")
            match value:
                case v if v is None and opt.value.parseas == "bool":
                    continue  # cfg[ini_key] = "false"
                case None:
                    continue
                case s if isinstance(s, set):
                    cfg[ini_key] = ",".join([str(el) for el in sorted(list(s))])
                case _:
                    cfg[ini_key] = str(value)
        return cfg

    def __str__(self):
        return " ".join(map(shlex.quote, self.as_args()))


def run_command(
    command: List[str], timeout: Optional[int] = None
) -> Tuple[int, str, str]:
    """
    Executes a shell command and returns its output.
    Returns a tuple: (return code, stdout, stderr)
    """
    command_str = " ".join(map(shlex.quote, command))
    logging.debug(f"Running command: {command_str}")
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Timeout error ({timeout}s)"
    except FileNotFoundError:
        return -2, "", f"Command not found: {command[0]}"
    except Exception as e:
        return -3, "", f"Command error: '{str(e)}'"


def is_valid_country_code(code: str) -> bool:
    """Checks if a string looks like a two-letter uppercase country code."""
    return len(code) == 2 and all([c.isupper() for c in code])


class CountryRegistry:
    """Manages the list of countries available for reflector."""

    def __init__(self):
        self._code_to_name: Dict[str, str] = {}
        self._name_to_code: Dict[str, str] = {}

    def _parse_and_add(self, line: str):
        """Parses a single line from `reflector --list-countries` output using regex."""
        line = line.strip()
        if not line:
            return
        match = re.match(r"^(.+?)\s+([A-Z]{2})\s+\d+$", line)
        if match:
            name = match.group(1).strip()
            code = match.group(2)
            logging.debug(f"Parsed country: Name='{name}', Code='{code}'")
            self._code_to_name[code] = name
            self._name_to_code[name.lower()] = code
        else:
            logging.warning(f"Could not parse country line: '{line}'")

    def load_countries(self) -> bool:
        """Loads countries by running `reflector --list-countries`."""
        ecode, out, err = run_command([REFLECTOR_CMD, ROpt.LIST_COUNTRIES.value.long])
        if ecode != 0:
            messagebox.showerror(
                "Error",
                f"Failed to retrieve country list from reflector.\n\nError:\n{err}",
            )
            return False

        start_parsing = False
        for line in out.splitlines():
            if line.startswith("-----"):
                start_parsing = True
                continue
            if start_parsing:
                self._parse_and_add(line)

        if not self._code_to_name:
            messagebox.showerror("Error", "No countries parsed from reflector output.")
            return False
        # Add Worldwide manually
        if "Worldwide" not in self._name_to_code:
            logging.debug("Adding 'Worldwide (WW)' manually to country list.")
            self._code_to_name["WW"] = "Worldwide"
            self._name_to_code["worldwide"] = "WW"
        logging.info(f"Loaded {len(self._code_to_name)} countries.")
        return True

    def get_name(self, code: str) -> Optional[str]:
        """Gets the country name for a given country code (case-insensitive)."""
        return self._code_to_name.get(code)

    def get_code(self, name: str) -> Optional[str]:
        """Gets the country code for a given name (case-insensitive)."""
        return self._name_to_code.get(name.lower())

    def all(self) -> List[tuple[str, str]]:
        """Returns a list of all loaded Countries: `(code, name)."""

        def by_name(cc, name):
            return ("0" if cc == "WW" else "1") + name

        return sorted(list(self._code_to_name.items()), key=lambda c: by_name(*c))

    def codes(self) -> Set[str]:
        return set(self._code_to_name.keys())

    def names(self) -> Set[str]:
        return set(self._name_to_code.keys())  # lowercased!


# Config
def parse_comma_separated_set(
    s: str, valid_items=Set[str], modifier=lambda x: x
) -> Set[str]:
    parsed = set()
    for x in s.split(","):
        item = modifier(x.strip()) if x else None
        if item and item in valid_items:
            parsed.add(item)
        else:
            logging.warning(
                f"Parsing error: invalid set item '{x}' '({item})'. Skipping."
            )
            continue
    return parsed


def get_app_config_dir(app_name: str = APP_NAME.lower()) -> str:
    dir = os.environ.get(
        "XDG_CONFIG_HOME", os.path.join(os.path.expanduser("~"), ".config")
    )
    return os.path.join(dir, app_name)


def save_config(
    config: ReflectorConfig, guiconfig: GuiConfig, cfg_dir: str = None
) -> None:
    """
    Saves the current GUI and Reflector configuration to an INI file.
    """
    out_dir = cfg_dir if cfg_dir else get_app_config_dir()
    logging.debug(f"Attempting to save configuration to: {out_dir}")
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_dict({"GUI": guiconfig, "Reflector": config.as_cfg_dict()})
    logging.debug("Reflector config prepared for saving.")

    # Ensure the directory exists
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:
        logging.error(f"Could not create configuration directory {out_dir}: {e}")
    cfg_path = os.path.join(out_dir, CONFIG_FILE_NAME)
    try:
        with open(cfg_path, "w", encoding="utf-8") as f:
            parser.write(f)
        logging.info(f"Successfully saved configuration to {cfg_path}.")
    except IOError as e:
        logging.error(f"Could not write configuration file {cfg_path}: {e}")


def read_config_file(cfg_path: str) -> Optional[str]:
    logging.info(f"Attempting to read config from: {cfg_path}")
    if not os.path.exists(cfg_path):
        logging.info(f"Config file {cfg_path} not found.")
        return None
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logging.info(f"Config file {cfg_path} not found.")


def parse_value(
    opt: ROpt, val_str: str, registry: CountryRegistry, arg: str
) -> Optional[float | str | int | Set[str] | bool]:
    """
    Parses a string value based on the expected type defined in ROpt.
    """
    val: Any = None
    try:
        match opt.value.parseas:
            case "float":
                val = float(val_str)
            case "int":
                val = int(val_str)
            case "set":
                if val_str is None or not val_str.strip():
                    val = set()
                else:
                    match opt:
                        case ROpt.PROTOCOLS:
                            val = parse_comma_separated_set(
                                val_str, valid_items=PROTOCOLS, modifier=str.lower
                            )
                        case ROpt.COUNTRIES:
                            val = parse_comma_separated_set(
                                val_str,
                                valid_items=registry.names() | registry.codes(),
                                modifier=lambda c: registry.get_code(c.lower())
                                if len(c) > 2
                                else c.upper(),
                            )
                        case _:
                            logging.warning(
                                f"Unsupported set type for option '{arg}'. Skipping."
                            )
                            return None  # failure/skip
                logging.debug(f"Parsed set for '{arg}': {val}")
            case "bool":
                raise ValueError(
                    "Program error: flag should have been parsed at previous step."
                )
            case _:
                val = val_str
        return val
    except ValueError | Exception as e:
        logging.warning(
            f"Error parsing value for option '{arg}' ('{val_str}') "
            f"as {opt.value.parseas}: {e}. Skipping."
        )
        return None


def parse_main_config(
    input: str, registry: CountryRegistry
) -> (GuiConfig, ReflectorConfig):
    if input is None:
        logging.info(f"No {APP_NAME} config to parse.")
    parser = configparser.ConfigParser(
        interpolation=None, inline_comment_prefixes=("#", ";")
    )
    guiconfig: GuiConfig = {}
    config = ReflectorConfig()
    try:
        parser.read_string(input)
        valid_int_keys = set(DEFAULT_GUI_CONFIG.keys())
        if "GUI" in parser:
            for key, val_str in parser.items("GUI"):
                if key in valid_int_keys:
                    try:
                        guiconfig[key] = parser.getint("GUI", key)
                    except ValueError:
                        logging.warning(
                            f"Invalid integer value for GUI.{key}: '{val_str}'. Skipping"
                        )
                else:
                    guiconfig[key] = val_str
            logging.info(f"Parsed loaded [GUI] config: {guiconfig}")
        if "Reflector" in parser:
            for key in parser["Reflector"]:
                arg = f"--{key}"  # Config key maps to long option name
                match OPT_STRING_TO_ENUM.get(arg):
                    case None | ROpt.SAVE:
                        logging.warning(
                            f"Ignoring unknown or unsupported option '{arg}' in reflector config."
                        )
                        continue
                    case opt if opt.value.parseas == "bool":
                        val = parser.getboolean("Reflector", key)
                        logging.debug(f"Setting flag: '{key}': {val}")
                        if val:
                            config[opt] = val
                    case opt:
                        val_str = parser.get("Reflector", key)
                        parsed_value = parse_value(opt, val_str, registry, arg)
                        if parsed_value is not None:
                            config[opt] = parsed_value
            logging.info(f"Parsed loaded [Reflector] config: {config}")
    except configparser.Error | Exception as e:
        logging.warning(f"Error parsing {APP_NAME} config file: {e}.")
    return guiconfig, config


def parse_reflector_config(
    input: str, registry: CountryRegistry, defaults: ReflectorConfig = None
) -> ReflectorConfig:
    # Start with a copy of defaults to ensure mutable defaults (sets) are independent
    config = copy.deepcopy(defaults) if defaults else ReflectorConfig()
    if input is None:
        logging.warning("No config to parse, using defaults.")
        return config
    try:
        args = shlex.split(input, comments=True)
        logging.debug(f"Parsed reflector config args: '{args}'")
    except ValueError as e:
        logging.error(f"Error tokenizing reflector config: {e}.")
        return config
    i = 0
    while i < len(args):
        arg = args[i]
        skip = False
        match OPT_STRING_TO_ENUM.get(arg):
            case None | ROpt.SAVE:
                logging.warning(
                    f"Ignoring unknown or unsupported option '{arg}' in reflector config."
                )
                skip = True
            case opt if opt.value.parseas == "bool":
                config[opt] = True
                logging.debug(f"Set flag: {opt.value.long}")
                skip = True
            case _ if i + 1 >= len(args) or args[i + 1] in OPT_STRING_TO_ENUM:
                logging.warning(
                    f"Option '{arg}' in reflector config file expects a value, but none was found. Skipping."
                )
                skip = True
            case _ if args[i + 1].startswith("-") and args[i + 1] in OPT_STRING_TO_ENUM:
                logging.warning(
                    f"Option '{arg}' in reflector config string expects a value, but found another option '{args[i + 1]}'. Skipping."
                )
                skip = True
            case opt:
                match parse_value(opt, args[i + 1], registry, arg):
                    case None:
                        skip = True
                    case val:
                        config[opt] = val
        if skip:
            i += 1
            continue
        i += 2
    logging.info(f"Parsed system Reflector config: {config}")
    return config


# /config


def get_current_country_code() -> Optional[str]:
    """
    Attempts to determine the user's country code.
    Returns the first valid code found or None.
    """
    GET_COUNTRY_COMMANDS = [
        (["show-location-info", "country"], None),
        (["curl", "-sf", "https://ipapi.co/country/"], DEFAULT_NET_TIMEOUT),
        (["curl", "-sf", "https://ifconfig.co/country-iso"], DEFAULT_NET_TIMEOUT),
    ]
    GET_IP_COMMANDS = [
        ["dig", "-4", "TXT", "+short", "o-o.myaddr.l.google.com", "@ns1.google.com"],
        ["curl", "-sf", "https://ipinfo.io/ip"],
        ["dig", "TXT", "+short", "o-o.myaddr.l.google.com", "@ns1.google.com"],
    ]
    GET_COUNTRY_BY_IP = [
        (["geoiplookup"], r"\:\s+([A-Z]{2}),\s"),
        (["whois"], r"country\:\s+([A-Z]{2})\s"),
    ]

    def get_current_ip() -> Optional[ipaddress.IPv4Address | ipaddress.IPv6Address]:
        for cmd in GET_IP_COMMANDS:
            ec, out, err = run_command(cmd, timeout=DEFAULT_NET_TIMEOUT)
            if ec == 0 and out:
                ips = out.strip(whitespace + "'\"")
                try:
                    ip = ipaddress.ip_address(ips)
                except ValueError:
                    logging.warning(f"Error parsing IP address '{ips}'")
                    continue
                logging.info(f"IP determined: {ip}")
                return ip
        return None

    # get country code
    for cmd, timeout in GET_COUNTRY_COMMANDS:
        ec, out, err = run_command(cmd, timeout=timeout)
        code = out.strip()
        if ec == 0 and is_valid_country_code(code):
            return code

    ip = get_current_ip()
    if ip:
        for cmd, reg in GET_COUNTRY_BY_IP:
            ec, out, err = run_command(cmd + [str(ip)], timeout=DEFAULT_NET_TIMEOUT)
            if ec == 0 and out:
                code_match = re.search(reg, out)
                if code_match:
                    code = code_match.group(1)
                    if is_valid_country_code(code):
                        return code

    # locale fallback
    lang_code, encoding = locale.getlocale(category=locale.LC_TIME)
    if lang_code:
        parts = lang_code.split("_")
        if len(parts) > 1 and is_valid_country_code(parts[1]):
            return parts[1]

    logging.warning("Could not determine local country code.")
    return None


def gui_err(header: str, msg: str):
    logging.error(msg)
    messagebox.showerror(header, msg)


def fetch_arch_mirrorlist(url: str = ARCH_MIRRORLIST_URL) -> Optional[str]:
    """Downloads the full mirrorlist from Arch Linux website."""
    try:
        # Download the content
        with urllib.request.urlopen(url, timeout=DEFAULT_NET_TIMEOUT) as response:
            if response.getcode() == 200:
                content = response.read().decode("utf-8")
                logging.info("Successfully downloaded full mirrorlist.")
                return content
            else:
                logging.warning(
                    "Failed to download full mirrorlist."
                    f"HTTP Status Code: {response.getcode()}"
                )

    except urllib.error.URLError as e:
        logging.warning(f"Network error fetching mirrorlist: {e}")
    except Exception as e:
        logging.warning(f"Unexpected error fetching the mirrorlist: {e}")
    return None


def split_server(s: str, prefix: str) -> Optional[str]:
    if s.startswith(prefix):
        server = s[len(prefix) :].strip()
        if len(server) > 6:  # Required minimum e.g. `FTP://`
            return server
        else:
            return None
    else:
        return None


def build_server_country_map(full_mirrorlist: str) -> Dict[str, str]:
    """
    Builds a dictionary mapping server URLs to their corresponding country codes.
    from the full Arch Linux mirrorlist.
    """
    server_to_country: Dict[str, str] = {}
    current_country = ""  # Default header before any country block
    for line in map(str.strip, full_mirrorlist.splitlines()):
        if (
            line == "##"
            or line.startswith("## Arch Linux")
            or line.startswith("## Generated")
        ):
            continue
        if line.startswith("## "):
            current_country = line[3:].strip()
        else:
            match split_server(line, COMMENTED_SERVER_PREFIX):
                case None:
                    pass
                case server:
                    server_to_country[server] = current_country
    logging.info(f"Built server-to-country map with {len(server_to_country)} entries.")
    return server_to_country


def add_country_names_to_mirrors(
    mirrorlist: str, registry: CountryRegistry, server_to_country: Dict[str, str]
) -> str:
    """
    Adds country comment headers above corresponding servers
    in the generated mirrorlist, using the pre-built map for reference.
    """
    output_lines = []
    last_country: Optional[str] = None
    for line in map(str.strip, mirrorlist.splitlines()):
        match split_server(line, SERVER_PREFIX):
            case None:
                # This is a non-server line (comment, blank line, etc.)
                output_lines.append(line)
                last_country = None
            case server:
                cur_country = server_to_country.get(server)
                if cur_country != last_country:
                    if cur_country is None:
                        logging.warning(f"Server not in country map: {server}")
                        output_lines.append("\n## UNKNOWN")
                    else:
                        output_lines.append(f"\n## {cur_country}")
                output_lines.append(line)
                last_country = cur_country
    return "\n".join(output_lines)


def save_to_default(mirrorlist_content: str, window: tk.Toplevel):
    """
    Saves the mirrorlist content to the default system location using pkexec.
    """
    pkexec_path = shutil.which(PKEXEC_CMD)
    if not pkexec_path:
        gui_err(
            "Privilege elevation error",
            "pkexec command not found.\n"
            "Cannot save to system directory without privilege elevation.\n"
            "Please use 'Save As...' or configure pkexec.",
        )
        return

    tmp_save_path = None
    try:
        # 1. Backup the current mirrorlist using pkexec
        if os.path.exists(MIRRORLIST_PATH):
            backup_cmd = [
                pkexec_path,
                "cp",
                "-a",
                MIRRORLIST_PATH,
                BACKUP_MIRRORLIST_PATH,
            ]
            logging.info(f"Running backup command: {' '.join(backup_cmd)}")
            code, _, stderr = run_command(backup_cmd)
            if code != 0:
                gui_err(
                    "Error",
                    f"Failed to create backup of old mirrorlist to '{BACKUP_MIRRORLIST_PATH}'\n\n"
                    f"Command executed: {' '.join(backup_cmd)}\n\n"
                    f"{stderr}",
                )
                return
        # 2. Write the new mirrorlist content to a temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, encoding="utf-8"
        ) as tmp_file:
            tmp_file.write(mirrorlist_content)
            tmp_save_path = tmp_file.name

        # 3. Use pkexec to copy the temporary file to the final destination
        save_cmd = [pkexec_path, "cp", tmp_save_path, MIRRORLIST_PATH]
        logging.info(f"Running save command: {' '.join(save_cmd)}")
        code, _, stderr = run_command(save_cmd)
        if code != 0:
            gui_err(
                "Error copying temporary file",
                f"Failed to save new mirrorlist to '{MIRRORLIST_PATH}'\n\n"
                f"Command executed: {' '.join(save_cmd)}\n\n"
                f"{stderr}",
            )
            # Attempt to restore the backup if save failed
            if os.path.exists(BACKUP_MIRRORLIST_PATH):
                restore_cmd = [
                    pkexec_path,
                    "cp",
                    "-a",
                    BACKUP_MIRRORLIST_PATH,
                    MIRRORLIST_PATH,
                ]
                logging.info(f"Attempting to restore backup: {' '.join(restore_cmd)}")
                _, _, _ = run_command(restore_cmd)
            return  # Stop after failure

        msg = f"New mirrorlist saved to {MIRRORLIST_PATH}."
        messagebox.showinfo("Success", msg)
        logging.info(msg)
        window.destroy()  # Close the preview window

    except Exception as e:
        gui_err("Error", f"An unexpected error occurred during save: {e}")
    finally:
        if tmp_save_path and os.path.exists(tmp_save_path):
            try:
                os.remove(tmp_save_path)
            except OSError as e:
                logging.warning(
                    f"Could not remove temporary save file {tmp_save_path}: {e}"
                )


def save_as(mirrorlist: str, window: tk.Toplevel):
    """
    Opens a save file dialog and saves the mirrorlist content to the selected location.
    """
    filepath = filedialog.asksaveasfilename(
        defaultextension="",
        filetypes=[
            ("Mirrorlist files", "*"),
            ("Conf files", "*.conf"),
            ("Text files", "*.txt"),
            ("All files", "*.*"),
        ],
        title="Save Mirrorlist As...",
        initialfile="mirrorlist",
        parent=window,
    )
    if not filepath:
        return  # User cancelled
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(mirrorlist)
        messagebox.showinfo("Success", f"Mirrorlist saved to '{filepath}'")
    except Exception as e:
        gui_err("Saving error", f"Failed to save mirrorlist to '{filepath}': {e}")


def show_preview_window(mirrorlist: str):
    """
    Displays the mirrorlist in a modal window with save options.
    """
    preview_window = tk.Toplevel()
    preview_window.title("Generated Mirrorlist Preview")
    preview_window.geometry("700x550")  # Slightly taller for buttons
    preview_window.minsize(500, 400)

    # Make the window modal
    preview_window.grab_set()
    preview_window.transient()

    text_area = scrolledtext.ScrolledText(
        preview_window, wrap=tk.WORD, width=80, height=15
    )
    text_area.pack(padx=10, pady=(10, 0), fill=tk.BOTH, expand=True)
    text_area.insert(tk.INSERT, mirrorlist)
    text_area.configure(state="disabled")  # read-only

    button_frame = ttk.Frame(preview_window)
    button_frame.pack(pady=10, padx=10, fill=tk.X, side=tk.RIGHT)

    def make_button(text, command, grid_args):
        btn = ttk.Button(
            button_frame,
            text=text,
            command=command,
        )
        btn.grid(row=0, **grid_args)

    make_button(
        text=f"Save to {MIRRORLIST_PATH}",
        command=lambda: save_to_default(mirrorlist, preview_window),
        grid_args={"column": 0, "padx": (0, 5), "sticky": tk.W},
    )

    make_button(
        text="Save As...",
        command=lambda: save_as(mirrorlist, preview_window),
        grid_args={"column": 1, "padx": 5},
    )

    make_button(
        text="Close",
        command=preview_window.destroy,
        grid_args={"column": 2, "padx": (5, 0), "sticky": tk.E},
    )

    preview_window.wait_window()


def wait(master, q):
    while q.empty():
        master.update()
        master.after(50)
    return q.get(block=True)


def background(root, tasks, resps, spinchan):
    while True:
        task = tasks.get()
        match task:
            case MsgKind.Done:
                return
            case (MsgKind.Start, "load_countries", registry):
                res = registry.load_countries()
            case (MsgKind.Start, "read_configs", registry):
                spinchan.put((MsgKind.Msg, ("Reading Reflector config...")))
                config = parse_reflector_config(
                    input=read_config_file(REFLECTOR_CONF_PATH),
                    registry=registry,
                    defaults=ReflectorConfig.default(),
                )
                spinchan.put((MsgKind.Msg, ("Reading main config...")))
                loaded_guiconfig, loaded_config = parse_main_config(
                    input=read_config_file(
                        os.path.join(get_app_config_dir(), CONFIG_FILE_NAME)
                    ),
                    registry=registry,
                )
                guiconfig = DEFAULT_GUI_CONFIG
                guiconfig.update(loaded_guiconfig)
                config.update(loaded_config)
                spinchan.put((MsgKind.Msg, ("Determining local country code...")))
                if not config[ROpt.COUNTRIES]:
                    local_code = get_current_country_code()
                    if local_code:
                        name = registry.get_name(local_code)
                        if name is not None:
                            logging.info(
                                f"Auto-selecting local country: {name} ({local_code})"
                            )
                            config[ROpt.COUNTRIES] = {local_code}
                res = (config, guiconfig)
            case (MsgKind.Start, "run_reflector", cmd):
                res = run_command(cmd)
            case (MsgKind.Start, "build_server_map"):
                mirrorlist = fetch_arch_mirrorlist()
                res = build_server_country_map(mirrorlist) if mirrorlist else None
            case task:
                gui_err("Error", f"Unknow background task: {task}")
        resps.put(res)


def run_reflector_and_show_preview(
    master: tk.Toplevel,
    config: ReflectorConfig,
    registry: CountryRegistry,
    server_map: Dict[str, str],
    tasks,
    resps,
):
    """
    Builds the command basesd on current config, runs reflector,
    processes the output, and shows the preview window.
    """
    reflector_cmd = [REFLECTOR_CMD]
    reflector_cmd.extend(config.as_args())
    spinner = SpinningMailbox(master)
    spinner.start()
    spinner.chan.put((MsgKind.Start, ("Running Reflector", "")))
    if server_map == {}:
        spinner.chan.put((MsgKind.Msg, ("Fetching full mirrorlist...")))
        server_map.update(wait(master, resps))  # should be prefetched
    spinner.chan.put((MsgKind.Msg, ("Waiting for Reflector to finish...")))
    tasks.put((MsgKind.Start, "run_reflector", reflector_cmd))
    try:
        ec, mirrorlist, err = wait(master, resps)
        if ec != 0:
            gui_err(
                "Reflector error",
                f"Reflector command failed (code: {ec}).\n\n"
                f"Command:\n'{reflector_cmd}'\n\n"
                f"Error Output:\n{err}",
            )
            return server_map
    except Exception as e:
        gui_err("Error", f"An unexpected error occurred while running reflector: {e}")
        return server_map
    finally:
        spinner.chan.put((MsgKind.Done, None))

    if "Server = " not in mirrorlist:
        gui_err(
            "Reflector returned no mirrors",
            """
            No mirrors found! Reflector returned no servers.
            You may need to change option values:
              * Select different or more countries.
              * Use a larger --age value.
              * Check selected protocols.
              * Verify any custom parameters.""",
        )
        return
    if server_map is not None:
        mirrorlist = add_country_names_to_mirrors(mirrorlist, registry, server_map)
    show_preview_window(mirrorlist)
    return server_map


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except tk.TclError:
        logging.info("Clam theme not available, using default.")
    root.title(WINDOW_TITLE)
    root.resizable(False, False)
    tasks, resps = queue.SimpleQueue(), queue.SimpleQueue()
    spinner = SpinningMailbox(root)
    back = Thread(
        target=background, args=[root, tasks, resps, spinner.chan], daemon=True
    )
    back.start()
    spinner.start()

    registry = CountryRegistry()
    spinner.chan.put(
        (MsgKind.Start, ("Downloading", "Downloading list of countries..."))
    )
    tasks.put((MsgKind.Start, "load_countries", registry))
    countries_loaded = wait(root, resps)
    if not countries_loaded:
        gui_err("Error", "Exiting due to failure loading countries.")
        return

    spinner.chan.put((MsgKind.Msg, ("Reading configs...")))
    tasks.put((MsgKind.Start, "read_configs", registry))
    config, guiconfig = wait(root, resps)
    spinner.chan.put((MsgKind.Done, None))

    server_map = {}  # returned on first reflector launch
    tasks.put((MsgKind.Start, "build_server_map"))  # prefetching mirrorlist
    root.deiconify()
    # --- Main Frame ---
    frame = ttk.Frame(root, padding=5)
    frame.pack()
    # --- Country Selection ---
    country_frame = ttk.LabelFrame(frame, text="Countries")
    country_frame.pack(fill=tk.X, expand=False)

    def labeled_control(
        text: str,
        control: ttk.Widget,
        frame: ttk.Frame,
        config_par: ROpt,
        rown: int,
        control_pars: Dict = {},
    ):
        def update_str(par: str, text: str):
            config[par] = text

        ttk.Label(frame, text=text).grid(row=rown, sticky=tk.W, pady=2)
        statevar = tk.StringVar(value=config[config_par])
        control(
            frame,
            textvariable=statevar,
            **control_pars,
        ).grid(row=rown, column=1, sticky=tk.W + tk.E, padx=5)
        statevar.trace_add(
            "write", lambda a, b, c, v=statevar: update_str(config_par, v.get())
        )
        return statevar, rown + 1

    country_list = registry.all()
    country_vars: Dict[str, tk.BooleanVar] = {}
    columns = guiconfig["columns"]
    rows = (len(country_list) // columns) + (len(country_list) % columns > 0)
    for i, (cc, cname) in enumerate(country_list):
        var = tk.BooleanVar()
        var.set(cc in config[ROpt.COUNTRIES])
        country_vars[cc] = var
        ttk.Checkbutton(
            country_frame,
            text=cname,
            variable=var,
            command=lambda c=cc, v=var: config.toggle_set_el(ROpt.COUNTRIES, c, v.get()),
        ).grid(row=i % rows, column=i // rows, sticky=tk.W, padx=5, pady=2)

    # --- Options panel ---
    options_frame = ttk.LabelFrame(frame, text="Options")
    options_frame.pack(fill=tk.X)
    options_frame.columnconfigure(0, weight=1)
    options_frame.columnconfigure(1, weight=1)
    # -- Protocols --
    subframe_l = ttk.Frame(options_frame)
    subframe_l.grid(row=0, column=0)

    rown = 0
    ttk.Label(subframe_l, text="Include mirrors:").grid(
        row=rown, column=0, sticky=tk.W, pady=2
    )
    proto_frame = ttk.Frame(subframe_l)
    proto_frame.grid(row=rown, column=1)
    proto_vars: Dict[str, tk.BooleanVar] = {}

    for proto in PROTOCOLS:
        v = tk.BooleanVar()
        v.set(proto in config[ROpt.PROTOCOLS])
        proto_vars[proto] = v
        ttk.Checkbutton(
            proto_frame,
            text=proto,
            variable=v,
            command=lambda p=proto, v=v: config.toggle_set_el(
                ROpt.PROTOCOLS, p, v.get()
            ),
        ).pack(side=tk.LEFT, padx=(0, 10))
    rown += 1

    # -- Sort --
    sort_var, rown = labeled_control(
        "Sort by:",
        ttk.Combobox,
        subframe_l,
        ROpt.SORT,
        rown,
        {"values": SORT_OPTIONS, "state": "readonly"},
    )
    # -- Age --
    age_var, rown = labeled_control(
        "Max sync age (hours):",
        ttk.Spinbox,
        subframe_l,
        ROpt.AGE,
        rown,
        {"from_": 0, "to_": 24 * 365},
    )

    subframe_r = ttk.Frame(options_frame)
    subframe_r.grid(row=0, column=1)
    rown = 0

    # -- Number of Mirrors --
    number_var, rown = labeled_control(
        "Max number of mirrors:",
        ttk.Spinbox,
        subframe_r,
        ROpt.NUMBER,
        rown,
        {"from_": 1, "to_": 1000},
    )
    # -- Timeout --
    timeout_var, rown = labeled_control(
        "Donwload timeout (s):",
        ttk.Spinbox,
        subframe_r,
        ROpt.DOWNLOAD_TIMEOUT,
        rown,
        {"from_": 1, "to_": 600},
    )
    # -- Threads --
    threads_var, rown = labeled_control(
        "Threads:",
        ttk.Spinbox,
        subframe_r,
        ROpt.THREADS,
        rown,
        {"from_": 1, "to_": os.cpu_count() * 4},
    )
    # -- Custom Params --
    subframe_b = ttk.Frame(options_frame)
    subframe_b.grid(row=1, column=0, columnspan=2, sticky=tk.E + tk.W)
    subframe_b.columnconfigure(1, weight=1)
    optionals_var, _ = labeled_control(
        "Custom reflector parameters:",
        ttk.Entry,
        subframe_b,
        ROpt.CUSTOM_ARGS,
        0,
    )

    def run():
        nonlocal server_map # mutated in background thread!
        run_reflector_and_show_preview(root, config, registry, server_map, tasks, resps)

    run_button = ttk.Button(frame, text="Run Reflector and Preview", command=run)
    run_button.pack(side=tk.RIGHT, pady=(10, 0), ipadx=5)
    run_button.focus_set()
    root.mainloop()
    logging.debug(f"Saving final config:\n [GUI]: {guiconfig}\n [Reflector]: {config}")
    save_config(config=config, guiconfig=guiconfig)


if __name__ == "__main__":
    if "--debug" in sys.argv:
        logging.getLogger().setLevel(logging.DEBUG)
    main()
