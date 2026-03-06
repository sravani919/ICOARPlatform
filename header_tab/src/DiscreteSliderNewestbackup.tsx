// DiscreteSlider.tsx (FULL UPDATED — collapsible sidebar rail + classic Material icons)
// ✅ Sidebar mode: default COLLAPSED (hamburger + icons only)
// ✅ Click hamburger -> expands (logo + icons + labels)
// ✅ Sends { value, sidebarOpen } to Streamlit in sidebar mode
// ✅ Uses ONLY Material-UI v4 icons that exist in @material-ui/icons
//
// Build steps (inside header_tab/):
//   npm i @material-ui/icons
//   npm run build

import React, { ReactNode } from "react";
import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
  ComponentProps,
} from "streamlit-component-lib";
import { Box, Tabs, Tab } from "@material-ui/core";
import icoarLogo from "./icoar_logo.png";

// Material-UI icons (v4-safe)
import HomeIcon from "@material-ui/icons/Home";
import CloudUploadIcon from "@material-ui/icons/CloudUpload"; // upload symbol for Data Collection
import BuildIcon from "@material-ui/icons/Build";
import AssessmentIcon from "@material-ui/icons/Assessment";
import BarChartIcon from "@material-ui/icons/BarChart";
import ImageIcon from "@material-ui/icons/Image";
import AndroidIcon from "@material-ui/icons/Android";
import StarsIcon from "@material-ui/icons/Stars";
import PersonIcon from "@material-ui/icons/Person";

type Mode = "embedded" | "header" | "sidebar";

interface State {
  activeStep: number; // 0..8
  menuOpen: boolean; // embedded dropdown
  lastDefault: number;
  lastMenuOpen: boolean;

  // ✅ sidebar rail
  sidebarOpen: boolean;
  lastSidebarOpen: boolean;
}

const ORANGE = "#ff8c00";
const INACTIVE = "#6b7280";

const menuOptions = [
  "Data Collection", // 1
  "Pre-processing", // 2
  "Text Analysis", // 3
  "Visualization", // 4
  "Multi-media Analysis", // 5
  "AI Assistant", // 6
  "AI-Assisted Features", // 7
];

const ACCOUNT_VALUE = 8;

const iconMap: Record<number, React.ReactNode> = {
  0: <HomeIcon style={{ fontSize: 20 }} />,
  1: <CloudUploadIcon style={{ fontSize: 20 }} />, // ✅ upload icon
  2: <BuildIcon style={{ fontSize: 20 }} />,
  3: <AssessmentIcon style={{ fontSize: 20 }} />,
  4: <BarChartIcon style={{ fontSize: 20 }} />,
  5: <ImageIcon style={{ fontSize: 20 }} />,
  6: <AndroidIcon style={{ fontSize: 20 }} />,
  7: <StarsIcon style={{ fontSize: 20 }} />,
  8: <PersonIcon style={{ fontSize: 20 }} />,
};

class DiscreteSlider extends StreamlitComponentBase<State> {
  public constructor(props: ComponentProps) {
    super(props);

    const initial = Number(this.props.args?.default ?? 0);
    const safeInitial = Number.isFinite(initial) ? initial : 0;

    // ✅ default collapsed unless Python passes sidebarOpen=true
    const initialSidebarOpen =
      typeof this.props.args?.sidebarOpen === "boolean"
        ? (this.props.args.sidebarOpen as boolean)
        : false;

    this.state = {
      activeStep: safeInitial,
      menuOpen: false,
      lastDefault: safeInitial,
      lastMenuOpen: false,

      sidebarOpen: initialSidebarOpen,
      lastSidebarOpen: initialSidebarOpen,
    };
  }

  private iconColor = (active: boolean) => (active ? ORANGE : INACTIVE);

  /** Resize AFTER render so it never clips inside Streamlit iframe */
  private setDynamicHeight = () => {
    const mode = (this.props.args?.mode as Mode) ?? "header";

    requestAnimationFrame(() => {
      if (mode === "header") {
        Streamlit.setFrameHeight(60);
        return;
      }

      if (mode === "sidebar") {
        // expanded needs more height; collapsed is smaller but keep safe height so it fills column
        // set a stable height so it looks "full sidebar"
        Streamlit.setFrameHeight(980);
        return;
      }

      // embedded mode
      if (this.state.menuOpen) {
        Streamlit.setFrameHeight(60 + (menuOptions.length + 2) * 48 + 12);
      } else {
        Streamlit.setFrameHeight(110);
      }
    });
  };

  componentDidMount(): void {
    Streamlit.setComponentReady();
    this.setDynamicHeight();
  }

  componentDidUpdate(): void {
    const mode = (this.props.args?.mode as Mode) ?? "header";

    // embedded dropdown height sync
    if (mode === "embedded") {
      if (this.state.lastMenuOpen !== this.state.menuOpen) {
        this.setDynamicHeight();
        this.setState({ lastMenuOpen: this.state.menuOpen });
        return;
      }
    }

    // sidebar open/close height sync
    if (mode === "sidebar") {
      if (this.state.lastSidebarOpen !== this.state.sidebarOpen) {
        this.setDynamicHeight();
        this.setState({ lastSidebarOpen: this.state.sidebarOpen });
        return;
      }
    }

    // Sync default from Python when page changes
    const nextDefaultRaw = Number(this.props.args?.default ?? 0);
    const nextDefault = Number.isFinite(nextDefaultRaw) ? nextDefaultRaw : 0;

    // Sync sidebarOpen from Python (so refresh doesn't flip)
    const nextSidebarOpen =
      typeof this.props.args?.sidebarOpen === "boolean"
        ? (this.props.args.sidebarOpen as boolean)
        : this.state.sidebarOpen;

    const defaultChanged = nextDefault !== this.state.lastDefault;
    const sidebarChangedFromPy = nextSidebarOpen !== this.state.sidebarOpen;

    if (defaultChanged || sidebarChangedFromPy) {
      this.setState({
        activeStep: nextDefault,
        menuOpen: false,
        lastDefault: nextDefault,
        lastMenuOpen: false,
        sidebarOpen: nextSidebarOpen,
        lastSidebarOpen: nextSidebarOpen,
      });
    }
  }

  private emitSidebarValue = (value: number, sidebarOpen: boolean) => {
    // ✅ In sidebar mode we emit an object so Python can resize columns
    Streamlit.setComponentValue({ value, sidebarOpen });
  };

  private handleChange = (newValue: number) => {
    const mode = (this.props.args?.mode as Mode) ?? "header";

    this.setState(
      { activeStep: newValue, menuOpen: false, lastMenuOpen: false },
      () => {
        this.setDynamicHeight();

        if (mode === "sidebar") {
          this.emitSidebarValue(newValue, this.state.sidebarOpen);
        } else {
          Streamlit.setComponentValue(newValue);
        }
      }
    );
  };

  private toggleMenu = () => {
    this.setState(
      (s) => ({ ...s, menuOpen: !s.menuOpen }),
      () => this.setDynamicHeight()
    );
  };

  private toggleSidebar = () => {
    const next = !this.state.sidebarOpen;
    this.setState({ sidebarOpen: next }, () => {
      this.setDynamicHeight();
      // keep current selection but update open state in Python
      this.emitSidebarValue(this.state.activeStep, next);
    });
  };

  public render = (): ReactNode => {
    const mode = (this.props.args?.mode as Mode) ?? "header";

    // =========================================================
    // ✅ SIDEBAR MODE (COLLAPSIBLE RAIL)
    // =========================================================
    if (mode === "sidebar") {
      const open = this.state.sidebarOpen;

      const railBtnStyle: React.CSSProperties = {
        width: 42,
        height: 42,
        borderRadius: 12,
        border: `1px solid ${open ? ORANGE : "#e5e7eb"}`,
        background: "#fff",
        cursor: "pointer",
        fontSize: 20,
        fontWeight: 900,
        lineHeight: "42px",
      };

      const itemStyle = (active: boolean): React.CSSProperties => ({
        padding: open ? "12px 12px" : "10px 10px",
        borderRadius: 12,
        cursor: "pointer",
        fontWeight: 900,
        color: active ? ORANGE : "#333",
        background: active ? "#fff4e6" : "transparent",
        border: active ? `1px solid ${ORANGE}` : "1px solid transparent",
        userSelect: "none",
        display: "flex",
        alignItems: "center",
        justifyContent: open ? "flex-start" : "center",
        gap: open ? 10 : 0,
      });

      return (
        <div style={{ width: "100%" }}>
          <div
            style={{
              background: "#ffffff",
              border: "1px solid #e6e6e6",
              borderRadius: 16,
              padding: 10,
              boxShadow: "0 2px 10px rgba(0,0,0,0.06)",
              height: "100%",
            }}
          >
            {/* Top: hamburger */}
            <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "6px 4px" }}>
              <button
                onMouseDown={(e) => e.preventDefault()}
                onClick={this.toggleSidebar}
                style={railBtnStyle}
                aria-label="Toggle sidebar"
                title={open ? "Collapse" : "Expand"}
                type="button"
              >
                {open ? "✕" : "☰"}
              </button>

              {open && (
                <>
                  <img src={icoarLogo} alt="ICOAR" style={{ height: 28 }} />
                  <div style={{ fontWeight: 900, color: ORANGE, fontSize: 18 }}>ICOAR</div>
                </>
              )}
            </div>

            <div style={{ height: 8 }} />

            {/* Home */}
            <div
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => this.handleChange(0)}
              style={itemStyle(this.state.activeStep === 0)}
              title="Home"
            >
              <span style={{ color: this.iconColor(this.state.activeStep === 0) }}>
                {iconMap[0]}
              </span>
              {open && <span>Home</span>}
            </div>

            <div style={{ height: 8 }} />

            {/* Menu items */}
            {menuOptions.map((label, i) => {
              const value = i + 1;
              const active = this.state.activeStep === value;
              return (
                <div
                  key={value}
                  onMouseDown={(e) => e.preventDefault()}
                  onClick={() => this.handleChange(value)}
                  style={{ ...itemStyle(active), marginBottom: 6 }}
                  title={label}
                >
                  <span style={{ color: this.iconColor(active) }}>{iconMap[value]}</span>
                  {open && <span>{label}</span>}
                </div>
              );
            })}

            {/* Account */}
            <div
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => this.handleChange(ACCOUNT_VALUE)}
              style={{ ...itemStyle(this.state.activeStep === ACCOUNT_VALUE), marginTop: 6 }}
              title="Account"
            >
              <span style={{ color: this.iconColor(this.state.activeStep === ACCOUNT_VALUE) }}>
                {iconMap[ACCOUNT_VALUE]}
              </span>
              {open && <span>Account</span>}
            </div>
          </div>
        </div>
      );
    }

    // =========================================================
    // EMBEDDED MODE (hamburger dropdown) — unchanged
    // =========================================================
    if (mode === "embedded") {
      return (
        <div style={{ width: "100%", position: "relative" }}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              background: "#ffffff",
              border: "1px solid #e6e6e6",
              borderRadius: 14,
              padding: "10px 12px",
              boxShadow: "0 2px 8px rgba(0,0,0,0.06)",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <button
                onMouseDown={(e) => e.preventDefault()}
                onClick={this.toggleMenu}
                style={{
                  width: 42,
                  height: 42,
                  borderRadius: 12,
                  border: `1px solid ${this.state.menuOpen ? ORANGE : "#e5e7eb"}`,
                  background: "#fff",
                  cursor: "pointer",
                  fontSize: 20,
                  fontWeight: 900,
                  lineHeight: "42px",
                }}
                aria-label="Open menu"
                title="Menu"
                type="button"
              >
                {this.state.menuOpen ? "✕" : "☰"}
              </button>

              <img src={icoarLogo} alt="ICOAR" style={{ height: 28 }} />
              <div style={{ fontWeight: 900, color: ORANGE, fontSize: 18 }}>ICOAR</div>
            </div>
            <div />
          </div>

          {this.state.menuOpen && (
            <div
              style={{
                position: "absolute",
                top: 62,
                left: 0,
                width: "100%",
                background: "#ffffff",
                border: "1px solid #e6e6e6",
                borderRadius: 14,
                overflow: "hidden",
                boxShadow: "0 10px 24px rgba(0,0,0,0.12)",
                zIndex: 999999,
              }}
            >
              <div
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => this.handleChange(0)}
                style={{
                  padding: "12px 14px",
                  cursor: "pointer",
                  fontWeight: 900,
                  display: "flex",
                  alignItems: "center",
                  gap: 10,
                  background: this.state.activeStep === 0 ? "#fff4e6" : "#fff",
                  color: this.state.activeStep === 0 ? ORANGE : "#333",
                  borderBottom: "1px solid #f0f0f0",
                }}
              >
                <span style={{ color: this.iconColor(this.state.activeStep === 0) }}>
                  {iconMap[0]}
                </span>
                Home
              </div>

              {menuOptions.map((label, i) => {
                const value = i + 1;
                const active = this.state.activeStep === value;
                return (
                  <div
                    key={value}
                    onMouseDown={(e) => e.preventDefault()}
                    onClick={() => this.handleChange(value)}
                    style={{
                      padding: "12px 14px",
                      cursor: "pointer",
                      fontWeight: 800,
                      color: active ? ORANGE : "#333",
                      borderBottom: value === 7 ? "none" : "1px solid #f0f0f0",
                      background: active ? "#fff4e6" : "#fff",
                      display: "flex",
                      alignItems: "center",
                      gap: 10,
                    }}
                  >
                    <span style={{ color: this.iconColor(active) }}>{iconMap[value]}</span>
                    {label}
                  </div>
                );
              })}

              <div
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => this.handleChange(ACCOUNT_VALUE)}
                style={{
                  padding: "12px 14px",
                  cursor: "pointer",
                  fontWeight: 900,
                  color: this.state.activeStep === ACCOUNT_VALUE ? ORANGE : "#666",
                  borderTop: "1px solid #f0f0f0",
                  background: this.state.activeStep === ACCOUNT_VALUE ? "#fff4e6" : "#fff",
                  display: "flex",
                  alignItems: "center",
                  gap: 10,
                }}
              >
                <span style={{ color: this.iconColor(this.state.activeStep === ACCOUNT_VALUE) }}>
                  {iconMap[ACCOUNT_VALUE]}
                </span>
                Account
              </div>
            </div>
          )}
        </div>
      );
    }

    // =========================================================
    // HEADER MODE (horizontal tabs) — unchanged
    // =========================================================
    return (
      <div style={{ height: "100%" }}>
        <Box
          style={{
            width: "100%",
            height: "60px",
            backgroundColor: "#f0f0f0",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            boxShadow: "0 4px 8px rgba(0, 0, 0, 0.08)",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              width: "14%",
              cursor: "pointer",
              paddingLeft: 10,
              gap: 8,
            }}
            onClick={() => this.handleChange(0)}
          >
            <img src={icoarLogo} alt="ICOAR Logo" style={{ height: "40px" }} />
            <div style={{ color: ORANGE, fontWeight: 900, fontSize: "20px" }}>ICOAR</div>
          </div>

          <div style={{ width: "76%", display: "flex", justifyContent: "center" }}>
            <Tabs
              value={String(this.state.activeStep)}
              onChange={(_, newValue) => this.handleChange(Number(newValue))}
              aria-label="ICOAR main tabs"
              TabIndicatorProps={{ style: { backgroundColor: ORANGE } }}
              style={{ width: "100%" }}
              variant="fullWidth"
            >
              {menuOptions.map((option, i) => {
                const key = i + 1;
                return (
                  <Tab
                    key={key}
                    value={String(key)}
                    label={option}
                    style={{
                      color: "#333",
                      fontWeight: 900,
                      fontSize: "13px",
                      height: "60px",
                      textTransform: "uppercase",
                    }}
                  />
                );
              })}
            </Tabs>
          </div>

          <div
            style={{
              width: "10%",
              textAlign: "center",
              color: this.state.activeStep === ACCOUNT_VALUE ? ORANGE : "#666",
              cursor: "pointer",
              fontWeight: 900,
            }}
            onClick={() => this.handleChange(ACCOUNT_VALUE)}
          >
            Account
          </div>
        </Box>
      </div>
    );
  };
}

export default withStreamlitConnection(DiscreteSlider);
