/**
 *
 * RecordVideo
 *
 */
import * as React from 'react';
import styled from 'styled-components/macro';
import { StyleConstants } from '../../../styles/StyleConstants';
import { PageWrapper } from '../../components/PageWrapper';
import { Helmet } from 'react-helmet-async';

import { useEffect, useState } from 'react';
import axios from 'axios';
import { Link } from '../../components/Link';
import { NavBar } from '../../components/NavBar';
import { LoadingOutlined } from '@ant-design/icons';

interface Props {}

export function FileList(props: Props) {
  const [file, setFile] = useState<any>(null);

  const [isLoading, setIsloading] = useState(false);

  const [datalist, setDatalist] = useState([]);

  useEffect(() => {
    const getFileList = async () => {
      setIsloading(true);

      const data = await axios.get('https://140.115.51.243/api/predict-list');

      console.log(data.data.filepath);
      setDatalist(data.data.filepath);

      console.log(data);
      setIsloading(false);
    };
    getFileList();
  }, []);

  return (
    <>
      <Helmet>
        <title>List</title>
        <meta
          name="description"
          content="A React Boilerplate application homepage"
        />
      </Helmet>
      <NavBar />
      <PageWrapperMain>
        {isLoading ? (
          <div
            className={
              'flex flex-col text-white text-7xl m-auto items-center justify-center'
            }
          >
            <LoadingOutlined className={'mb-12'} />
          </div>
        ) : (
          <div className={'flex h-full w-full bg-black justify-center'}>
            <div className="flex flex-col max-h-screen mt-8">
              <div className="text-center text-5xl font-bold font-sans text-white">
                File List
              </div>
              <div className="text-center text-md font-light font-sans text-red-500 mb-4">
                *Some files might take some time to predict, Thanks for your
                patience & understanding!
              </div>
              <div className="flex flex-col bg-white w-full overflow-y-auto">
                {datalist.map((item, index) => {
                  return (
                    <div className="flex flex-row text-orange-500 bg-blue-100 odd:bg-blue-200 text-sm font-light hover:bg-blue-500 hover:text-white">
                      <div className={'flex flex-1 px-4 py-1'}>{item}</div>
                      <a
                        className={
                          'flex h-full items-center justify-center mr-2 font-black cursor-pointer'
                        }
                        href={`https://140.115.51.243/api/download-original?filename=${item}.mp4`}
                      >
                        🗄️ Download
                      </a>
                      {/*<a*/}
                      {/*  className={*/}
                      {/*    'flex h-full items-center justify-center mr-2 font-black cursor-pointer'*/}
                      {/*  }*/}
                      {/*  href={`https://140.115.51.243/api/download-predict?filename=${item}.mp4`}*/}
                      {/*>*/}
                      {/*  ✨ Analyzed*/}
                      {/*</a>*/}
                      <Link
                        to={process.env.PUBLIC_URL + `/playback?title=${item}`}
                        className={'text-blue-500'}
                      >
                        <div
                          className={
                            'flex h-full items-center justify-center mr-4 font-black cursor-pointer'
                          }
                        >
                          ▶️ Analyze
                        </div>
                      </Link>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}
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
