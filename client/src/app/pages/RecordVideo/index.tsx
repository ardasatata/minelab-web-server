/**
 *
 * RecordVideo
 *
 */
import * as React from 'react';
import styled from 'styled-components/macro';
import { StyleConstants } from '../../../styles/StyleConstants';
import { PageWrapper } from '../../components/PageWrapper';
import { Logo } from '../../components/NavBar/Logo';
import { Nav } from '../../components/NavBar/Nav';
import { Helmet } from 'react-helmet-async';
import { Masthead } from '../HomePage/Masthead';
import { Features } from '../HomePage/Features';
import VideoRecorder from 'react-video-recorder';

interface Props {}

export function RecordVideo(props: Props) {
  return (
    <>
      <Helmet>
        <title>Home Page</title>
        <meta
          name="description"
          content="A React Boilerplate application homepage"
        />
      </Helmet>
      <NavBar />
      <PageWrapperMain>
        <div className={'flex h-full w-3/4'}>
          <VideoRecorder
            isReplayingVideo={false}
            showReplayControls={true}
            isOnInitially={true}
            // countdownTime={RECORDER_TIME_CONFIG.countdownTime}
            // timeLimit={RECORDER_TIME_CONFIG.timeLimit}
            mimeType="video/webm;codecs=vp8,opus"
            // onRecordingComplete={async (videoBlob) => {
            //   // Do something with the video...
            //   // setPrediction(['Uploading your video'])
            //   await submitVideo(videoBlob)
            //   sum(5, 6)
            //   console.log('videoBlob', videoBlob)
            // }}
            // onStartRecording={() => {
            //   setPrediction(['Recording video...'])
            // }}
            // onTurnOnCamera={
            //   () => {
            //     setPrediction(['Press Record First!'])
            //   }
            // }
          />
        </div>
        <div className={'h-full flex bg-white w-1/4'}>cok</div>
        {/*<PageWrapperMain>*/}
        {/*  <Masthead />*/}
        {/*  /!*<div className={'bg-white h-full flex flex-1'}>COK</div>*!/*/}
        {/*  /!*<Features />*!/*/}
        {/*</PageWrapperMain>*/}
      </PageWrapperMain>
    </>
  );
}

export function NavBar() {
  return (
    <Wrapper>
      <PageWrapper>
        <Logo />
        {/*<Nav />*/}
      </PageWrapper>
    </Wrapper>
  );
}

const Wrapper = styled.header`
  box-shadow: 0 1px 0 0 ${p => p.theme.borderLight};
  height: ${StyleConstants.NAV_BAR_HEIGHT};
  display: flex;
  position: fixed;
  top: 0;
  width: 100%;
  background-color: ${p => p.theme.background};
  z-index: 2;

  @supports (backdrop-filter: blur(10px)) {
    backdrop-filter: blur(10px);
    background-color: ${p =>
      p.theme.background.replace(
        /rgba?(\(\s*\d+\s*,\s*\d+\s*,\s*\d+)(?:\s*,.+?)?\)/,
        'rgba$1,0.75)',
      )};
  }

  ${PageWrapper} {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
`;

export const PageWrapperMain = styled.div`
  display: flex;
  margin: 0 auto;
  box-sizing: content-box;
  height: calc(100vh - ${StyleConstants.NAV_BAR_HEIGHT});
`;
