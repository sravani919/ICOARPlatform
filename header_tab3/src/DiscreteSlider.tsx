import { Box } from "@material-ui/core";
import { Tabs, Tab } from "@material-ui/core";
import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
  ComponentProps,
} from "streamlit-component-lib";
import React, { ReactNode } from "react";

interface State {
  activeStep: number;
}

const menu_options = [
  "Cyberbullying Image Analysis",
  "Meme Analysis",
  "Customized Image Analysis",
  "Cyberbullying Detection using GPT",
  "Deepfake Detection",
];

class DiscreteSlider extends StreamlitComponentBase<State> {
  public constructor(props: ComponentProps) {
    super(props);
    // ✅ default to first tab (1..N indexing)
    this.state = { activeStep: 1 };
  }

  public componentDidMount(): void {
    // ✅ stop overlap: ensure Streamlit iframe is tall enough
    Streamlit.setFrameHeight(110);

    // ✅ set default value on first mount (so Home.py has a choice immediately)
    Streamlit.setComponentValue(menu_options[0]);
  }

  public componentDidUpdate(): void {
    // keep stable on rerenders
    Streamlit.setFrameHeight(110);
  }

  public handleChange = (_event: React.ChangeEvent<{}>, newValue: number) => {
    this.setState({ activeStep: newValue });
    Streamlit.setComponentValue(menu_options[newValue - 1]);
  };

  public render = (): ReactNode => {
    return (
      <div style={{ width: "100%", display: "flex", justifyContent: "center" }}>
        <Box
          style={{
            width: "100%",
            maxWidth: "1100px",
            height: "72px",
            backgroundColor: "#f0f0f0",
            display: "flex",
            alignItems: "center",
            borderRadius: "10px",
            boxShadow: "0 4px 8px rgba(0, 0, 0, 0.06)",
            overflow: "hidden",
          }}
        >
          <Tabs
            value={this.state.activeStep}
            onChange={this.handleChange}
            aria-label="multi-media tabs"
            TabIndicatorProps={{
              style: { backgroundColor: "#ff8c00", height: 3 },
            }}
            variant="scrollable"
            scrollButtons="auto"
          >
            {menu_options.map((option, i) => {
              const key = i + 1;
              return (
                <Tab
                  key={key}
                  value={key}
                  label={option}
                  style={{
                    color: "#333",
                    fontWeight: 800,
                    fontSize: "14px",
                    height: "72px",
                    padding: "0 18px",
                    textTransform: "none",
                    minWidth: "auto",
                  }}
                />
              );
            })}
          </Tabs>
        </Box>
      </div>
    );
  };
}

export default withStreamlitConnection(DiscreteSlider);