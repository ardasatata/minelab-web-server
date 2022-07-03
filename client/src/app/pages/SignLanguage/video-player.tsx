/**
 *
 * RecordVideo
 *
 */
import * as React from 'react';
import { useEffect, useState, useRef } from 'react';
import styled from 'styled-components/macro';
import { StyleConstants } from '../../../styles/StyleConstants';
import { PageWrapper } from '../../components/PageWrapper';
import { Helmet } from 'react-helmet-async';

import { NavBar } from '../../components/NavBar';
import { useQuery } from '../../../hooks/useQuery';

import { Player } from 'video-react';
import ReactPlayer from 'react-player';
import { getJSON } from 'utils/getJson';
import axios from "axios";

interface Props {}

const OFFSET = 30

export function VideoPlayerSL(props: Props) {
  const query = useQuery();

  const title = query.get('title')

  const [prediction, setPrediction] = useState<string>('');

    const getPrediction = async () => {
      const res = await axios.get(`https://140.115.51.243/api/sentence-sl/${title}.mov.txt`);

      if(res.data.ok){
          setPrediction(res.data.prediction)
          console.log(res.data.prediction)
      }
    }

  // console.log(query.get('title'));
  // @ts-ignore
  const [socket, setSocket] = useState<any>(null);
  const [frame, setFrame] = useState<any>(0);
  const [img, setImg] = useState<string>('');
  const [data, setData] = useState<any>(null);

  const [labels, setLabels] = useState<Array<number>>([0]);

  // @ts-ignore
  useEffect(() => {
    getPrediction().then(()=>{

    })
  }, []);

  const play = () => {
    setPlaying(true);
  };

  useEffect(() => {
    // setTimeout(() => {
    //   play();
    // }, 2000);
  }, []);

  const refVideo = useRef(null);

  // refVideo.current.subscribeToStateChange(state => console.log(state));

  const [jsonData, setJsonData] = useState(null);
  const [playing, setPlaying] = useState(false);

  // useEffect(() => {
  //   getJSON(
  //     `https://140.115.51.243/api/predict/${query.get('title')}.json`,
  //     function (err, data) {
  //       if (err !== null) {
  //         alert('Something went wrong: ' + err);
  //       } else {
  //         console.log(data.length);
  //         setJsonData(data);
  //       }
  //     },
  //   );
  // }, []);

  return (
    <>
      <Helmet>
        <title>{`Evaluate ${title}`}</title>
        <meta
          name="description"
          content="A React Boilerplate application homepage"
        />
      </Helmet>
      <NavBar />
      <PageWrapperMain>
        <div className={'flex h-full w-full bg-black flex-1'}>
          <div
            className={
              'flex flex-col w-full h-full justify-center items-center'
            }
          >
            {/*<div className={'flex flex-row w-full'}>*/}
            {/*  <div*/}
            {/*    className={*/}
            {/*      'flex-1 text-white text-4xl py-4 transition ease-in-out bg-green-500 hover:bg-indigo-700 duration-300 items-center justify-center flex cursor-pointer'*/}
            {/*    }*/}
            {/*    onClick={play}*/}
            {/*  >*/}
            {/*    <PlayCircleOutlined className={'mr-2'} />*/}
            {/*    {'Re-PLAY'}*/}
            {/*  </div>*/}
            {/*  /!*<div*!/*/}
            {/*  /!*  className={*!/*/}
            {/*  /!*    'flex-1 text-white text-4xl py-4 transition ease-in-out bg-red-500 hover:bg-pink-700 duration-300 items-center justify-center flex cursor-pointer'*!/*/}
            {/*  /!*  }*!/*/}
            {/*  /!*  onClick={stop}*!/*/}
            {/*  /!*>*!/*/}
            {/*  /!*  <StopOutlined className={'mr-2'} />*!/*/}
            {/*  /!*  {'STOP'}*!/*/}
            {/*  /!*</div>*!/*/}
            {/*  <div*/}
            {/*    className={*/}
            {/*      'text-white text-4xl py-4 transition ease-in-out bg-blue-500 hover:bg-pink-700 duration-300 items-center justify-center flex cursor-pointer px-8'*/}
            {/*    }*/}
            {/*    onClick={toggleMenu}*/}
            {/*  >*/}
            {/*    <MenuOutlined className={'mr-2'} />*/}
            {/*    {'ANALYZE'}*/}
            {/*  </div>*/}
            {/*</div>*/}

            <div className="flex flex-1 justify-center max-w-7xl">
                <div
                  className={
                    'text-white text-3xl absolute mt-4 ml-4 transition ease-in-out bg-black p-2 z-50'
                  }
                >
                  {`Prediction : ${prediction}`}
                </div>

                <div
                  className={'bg-black'}
                  style={{
                    position: 'relative',
                    paddingLeft: 12,
                    minWidth: 960,
                  }}
                >
                  <ReactPlayer
                    className="absolute top-0 left-0"
                    url={`https://140.115.51.243:5001/predict-sl/${title}.mov`}
                    controls={true}
                    onProgress={state => {
                      const frameNum = Math.round(state.playedSeconds * 30);
                      // console.log(jsonData.length);
                      // console.log(frameNum);
                      if (frameNum < jsonData.length) {
                        setFrame(Math.round(frameNum) + OFFSET);
                      }
                    }}
                    // onPlay={() => console.log('play')}
                    progressInterval={1}
                    width="100%"
                    height="100%"
                    playbackRate={0.5}
                    playing={playing}
                    muted={true}
                  />
                </div>
              </div>

          </div>
        </div>
      </PageWrapperMain>
    </>
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
