import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
  ComponentProps
} from "streamlit-component-lib"
import React, { ReactNode } from "react"
import { Box } from "@material-ui/core"
import { Paper } from '@material-ui/core';

class Citations extends StreamlitComponentBase {
  public constructor(props: ComponentProps) {
    super(props)
    Streamlit.setComponentValue(0)
  }
  public render = (): ReactNode => {

    return (
      <Box
      style={{
        display: 'flex',
        flexWrap: 'wrap',
        width: '100%',
        height: '400px',
      }}
    >
      <Paper style={{ width: '40%', height: '100%', marginLeft: '5%', marginRight: '10%', backgroundColor: '#efefef' }} elevation={3}>
        <h3 style={{marginTop: '7%', marginLeft: '7%'}}>Citations</h3>
        <ul style={{marginRight: '3.5%'}}>
          <li>
          <a href=''>Using LLM's to mitigate the sread of hate speech</a>
        </li>
        <li>
          <a href=''>Using LICOAR tool for identiful sprea dof hate on social media</a>
        </li>
        </ul>
        </Paper>
      <Paper style={{ width: '40%', height: '100%', backgroundColor: '#efefef' }} elevation={3}>
        <h3 style={{marginTop: '7%', marginLeft: '7%'}}>About</h3>
        <p style={{marginLeft: '3.5%', marginRight: '3.5%', fontWeight: 'lighter'}}>
        Integrative Cyberinfrastructure for Online Abuse Research (ICOAR) is a scalable, adaptable, and user-friendly platform which advances research capability for researchers in both social science and computer science communities to leverage advanced machine learning methods for online abuse research.
        </p>
        {/* <ul>
          <li>
          <a href=''>Project 1</a>
        </li>
        <li>
          <a href=''>Project 2</a>
        </li>
        </ul> */}
        </Paper>
    </Box>
    )
  }
}

export default withStreamlitConnection(Citations)
